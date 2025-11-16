// Function: sub_16CB530
// Address: 0x16cb530
//
signed __int64 __fastcall sub_16CB530(signed __int64 a1, unsigned __int64 _RSI)
{
  signed __int64 result; // rax
  int v4; // edi
  unsigned __int64 v5; // rdx
  __int16 v6; // cx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rtt
  __int64 v9; // rcx
  char v10; // cl

  result = a1;
  if ( _RSI )
  {
    __asm { tzcnt   rdx, rsi }
    LOWORD(v4) = _RDX;
    if ( !(_DWORD)_RDX )
    {
      if ( _RSI == 1 )
        return result;
      goto LABEL_4;
    }
    v10 = _RDX;
    v4 = -(int)_RDX;
  }
  else
  {
    LOWORD(v4) = -64;
    v10 = 64;
  }
  _RSI >>= v10;
  if ( _RSI == 1 )
    return result;
LABEL_4:
  if ( !result )
  {
    v6 = 64;
    goto LABEL_21;
  }
  _BitScanReverse64(&v5, result);
  v6 = v5 ^ 0x3F;
  if ( (unsigned int)v5 != 0x3F )
  {
LABEL_21:
    LOWORD(v4) = v4 - v6;
    result <<= v6;
  }
  v8 = result;
  result /= _RSI;
  v7 = v8 % _RSI;
  if ( v8 % _RSI )
  {
    do
    {
      v9 = v7;
      LOWORD(v4) = v4 - 1;
      v7 *= 2LL;
      result *= 2LL;
      if ( v9 < 0 || v7 >= _RSI )
      {
        result |= 1uLL;
        v7 -= _RSI;
      }
    }
    while ( result >= 0 && v7 );
  }
  if ( v7 >= (_RSI >> 1) + (_RSI & 1) && !++result )
    return 0x8000000000000000LL;
  return result;
}
