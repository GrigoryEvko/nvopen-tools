// Function: sub_13A3700
// Address: 0x13a3700
//
__int64 __fastcall sub_13A3700(unsigned __int64 a1)
{
  __int64 result; // rax
  int v3; // ecx
  __int64 v4; // rax
  __int64 v5; // r8
  int v7; // edx

  if ( (a1 & 1) != 0 )
  {
    if ( (a1 >> 1) & ~(-1LL << (a1 >> 58)) )
    {
      __asm { tzcnt   rax, rax }
      return result;
    }
    return 0xFFFFFFFFLL;
  }
  v3 = *(_DWORD *)(a1 + 16);
  if ( !v3 )
    return 0xFFFFFFFFLL;
  v4 = 0;
  v5 = (unsigned int)(v3 - 1) >> 6;
  while ( 1 )
  {
    _RSI = *(_QWORD *)(*(_QWORD *)a1 + 8 * v4);
    v7 = v4;
    if ( v5 == v4 )
      break;
    if ( _RSI )
      goto LABEL_10;
    if ( (_DWORD)v5 + 1 == ++v4 )
      return 0xFFFFFFFFLL;
  }
  _RSI &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
  if ( !_RSI )
    return 0xFFFFFFFFLL;
LABEL_10:
  __asm { tzcnt   rax, rsi }
  return (unsigned int)((v7 << 6) + _RAX);
}
