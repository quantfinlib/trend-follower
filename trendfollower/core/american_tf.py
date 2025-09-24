# Copyright (c) 2025 Mohammadjavad Vakili, All rights reserved.

"""Implementation of the building blocks of an American Trend Following System."""


from collections.abc import Callable
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class MarketData(BaseModel):
    """Market data for a single time point."""

    model_config = {"frozen": True}  # Make immutable

    price: float = Field(gt=0, description="Current price.")
    fast_ema: float = Field(gt=0, description="Fast EMA.")
    slow_ema: float = Field(gt=0, description="Slow EMA.")
    atr: float = Field(gt=0, description="Average True Range.")


class StrategyParams(BaseModel):
    """Strategy parameters for American Trend Follower."""

    model_config = {"frozen": True}  # Make immutable

    entry_point_width: float = Field(gt=0, description="Entry point width as a multiple of ATR.")
    stop_loss_width: float = Field(gt=0, description="Stop loss width as a multiple of ATR.")
    risk_multiple: float = Field(gt=0, description="Risk multiple for position sizing.")


class AmericanTFState(BaseModel):
    """State of the American Trend Follower strategy."""

    model_config = {"frozen": True}  # Make immutable

    position: Literal[-1, 0, 1] = Field(
        default=0,
        description="Current position: 1 for long, -1 for short, 0 for neutral.",
    )
    position_size: float = Field(
        default=0,
        description="Size of the current position.",
    )
    stop_loss: float | None = Field(default=None, ge=0, description="Stop loss level.")
    entry_price: float | None = Field(
        default=None, gt=0,
        description="Entry price of the position.",
    )
    close_next_day: bool = Field(
        default=False,
        description="Whether to close the position the next day. True if stop-loss was hit but signal is still on.",
    )

    @model_validator(mode="after")
    def validate_position_consistency(self) -> Self:
        """Ensure position, position_size, stop_loss, and entry_price are consistent.

        Returns
        -------
        Self
            The validated state.

        Raises
        ------
        ValueError
            If any inconsistency is found.

        """
        invalid_neutral_position = (
            self.position == 0 and (
                (self.position_size != 0) or (self.stop_loss is not None) or (self.entry_price is not None)
            )
        )
        invalid_short_position = (
            self.position == -1 and (
                (self.position_size >= 0) or (self.stop_loss is None) or (self.entry_price is None)
            )
        )
        invalid_long_position = (
            self.position == 1 and (
                (self.position_size <= 0) or (self.stop_loss is None) or (self.entry_price is None)
            )
        )
        if invalid_neutral_position:
            err_msg = "For neutral position, position_size must be 0, stop_loss and entry_price must be None."
            raise ValueError(err_msg)
        if invalid_short_position:
            err_msg = "For short position, position_size must be negative, stop_loss and entry_price must be set."
            raise ValueError(err_msg)
        if invalid_long_position:
            err_msg = "For long position, position_size must be positive, stop_loss and entry_price must be set."
            raise ValueError(err_msg)
        return self


POSITION_HANDLER = Callable[[MarketData, StrategyParams, AmericanTFState], AmericanTFState]


def handle_neutral_position(
    market: MarketData, params: StrategyParams, state: AmericanTFState | None = None,
) -> AmericanTFState:
    """Update state of an American Trend Follower strategy when in a neutral position.

    Parameters
    ----------
    market : MarketData
        Current market data, containing price, fast EMA, slow EMA, and ATR.
    params : StrategyParams
        Strategy parameters, including entry point width, stop loss width, and risk multiple.
    state : AmericanTFState | None, optional
        Current state of the strategy. Not used in this handler, by default None.
        Not being used because in neutral state, previous state does not affect the decision.
        It is included for conformity with POSITION_HANDLER type.

    Returns
    -------
    AmericanTFState
        Updated state of the strategy.

    """
    signal_buffer: float = params.entry_point_width * market.atr
    stop_loss_buffer: float = params.stop_loss_width * market.atr
    position_size_magnitude: float = params.risk_multiple * market.price / market.atr
    # Long entry upon long signal
    if market.fast_ema > market.slow_ema + signal_buffer:
        return AmericanTFState(
            position=1,
            position_size=position_size_magnitude,
            stop_loss=market.price - stop_loss_buffer,
            entry_price=market.price,
            close_next_day=False,
        )
    # Short entry upon short signal
    if market.fast_ema < market.slow_ema - signal_buffer:
        return AmericanTFState(
            position=-1,
            position_size=-position_size_magnitude,
            stop_loss=market.price + stop_loss_buffer,
            entry_price=market.price,
            close_next_day=False,
        )
    # Remain neutral if no signal
    return AmericanTFState(
            position=0,
            position_size=0.0,
            stop_loss=None,
            entry_price=None,
            close_next_day=False,
        )


def handle_long_position(market: MarketData, params: StrategyParams, state: AmericanTFState) -> AmericanTFState:
    """Update state of an American Trend Follower strategy when in a long position.

    Parameters
    ----------
    market : MarketData
        Current market data, containing price, fast EMA, slow EMA, and ATR.
    params : StrategyParams
        Strategy parameters, including entry point width, stop loss width, and risk multiple.
    state : AmericanTFState
        Current state of the strategy.

    Returns
    -------
    AmericanTFState
        Updated state of the strategy.

    Raises
    ------
    ValueError
        If stop_loss is not set for an open position.

    """
    if state.stop_loss is None:
        msg = "stop_loss must be set for open positions"
        raise ValueError(msg)
    # Check if position was marked for closure from previous day
    if state.close_next_day:
        return AmericanTFState(
            position=0,
            position_size=0.0,
            stop_loss=None,
            entry_price=None,
            close_next_day=False,
        )

    signal_buffer = params.entry_point_width * market.atr
    stop_loss_buffer = params.stop_loss_width * market.atr

    if market.price < state.stop_loss:  # Stop-loss is breached
        if market.fast_ema <= market.slow_ema + signal_buffer:  # Signal is off, close position immediately
            return AmericanTFState(
                position=0,
                position_size=0.0,
                stop_loss=None,
                entry_price=None,
                close_next_day=False,
            )
        return state.model_copy(update={"close_next_day": True})
    # Stop-loss is not breached, update the stop-loss level
    new_stop_loss = max(state.stop_loss, market.price - stop_loss_buffer)
    return state.model_copy(update={"stop_loss": new_stop_loss})


def handle_short_position(market: MarketData, params: StrategyParams, state: AmericanTFState) -> AmericanTFState:
    """Update state of an American Trend Follower strategy when in a short position.

    Parameters
    ----------
    market : MarketData
        Current market data, containing price, fast EMA, slow EMA, and ATR.
    params : StrategyParams
        Strategy parameters, including entry point width, stop loss width, and risk multiple.
    state : AmericanTFState
        Current state of the strategy.

    Returns
    -------
    AmericanTFState
        Updated state of the strategy.

    Raises
    ------
    ValueError
        If stop_loss is not set for an open position.

    """
    if state.stop_loss is None:
        msg = "stop_loss must be set for open positions"
        raise ValueError(msg)
    # Check if position was marked for closure from previous day
    if state.close_next_day:
        return AmericanTFState(
            position=0,
            position_size=0,
            stop_loss=None,
            entry_price=None,
            close_next_day=False,
        )

    signal_buffer = params.entry_point_width * market.atr
    stop_loss_buffer = params.stop_loss_width * market.atr

    if market.price > state.stop_loss:  # Stop-loss is breached
        if market.fast_ema >= market.slow_ema - signal_buffer:  # Signal is off, close position immediately
            return AmericanTFState(
                position=0,
                position_size=0.0,
                stop_loss=None,
                entry_price=None,
                close_next_day=False,
            )
        # Signal is still on, keep position but mark for closure next day
        return state.model_copy(update={"close_next_day": True})
    # Stop-loss is not breached, update the stop-loss level
    new_stop_loss = min(state.stop_loss, market.price + stop_loss_buffer)
    return state.model_copy(update={"stop_loss": new_stop_loss})


class AmericanTrendFollower:
    """American-style trend following strategy.

    Parameters
    ----------
    entry_point_width : float
        Entry point width as a multiple of ATR.
    stop_loss_width : float
        Stop loss width as a multiple of ATR.
    risk_multiple : float
        Risk multiple for position sizing.

    Methods
    -------
    update_state(s, s_fast, s_slow, atr, state) -> AmericanTFState
        Update the state of the strategy given current market data and previous state.

    """

    def __init__(self, entry_point_width: float, stop_loss_width: float, risk_multiple: float) -> None:

        self.params = StrategyParams(
            entry_point_width=entry_point_width,
            stop_loss_width=stop_loss_width,
            risk_multiple=risk_multiple,
        )
        self.handlers: dict[Literal[-1, 0, 1], POSITION_HANDLER] = {
            0: handle_neutral_position,
            1: handle_long_position,
            -1: handle_short_position,
        }

    def update_state(
        self, s: float, s_fast: float, s_slow: float, atr: float, state: AmericanTFState
    ) -> AmericanTFState:
        """Update the state of the American Trend Follower strategy.

        Parameters
        ----------
        s : float
            Current price.
        s_fast : float
            Current fast EMA.
        s_slow : float
            Current slow EMA.
        atr : float
            Current ATR.
        state : AmericanTFState
            Current state of the strategy.

        Returns
        -------
        AmericanTFState
            Updated state of the strategy.

        """
        market = MarketData(price=s, fast_ema=s_fast, slow_ema=s_slow, atr=atr)
        handler = self.handlers[state.position]
        return handler(market, self.params, state)
