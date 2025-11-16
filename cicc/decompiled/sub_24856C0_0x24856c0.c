// Function: sub_24856C0
// Address: 0x24856c0
//
__int64 __fastcall sub_24856C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v7; // rbx
  __int64 v8; // r14

  v4 = 32 * a2;
  if ( v4 >> 5 != (8 * a4) >> 3 )
    return 0;
  v7 = a1;
  if ( a1 + v4 == a1 )
    return 1;
  while ( 1 )
  {
    v8 = *a3;
    if ( v8 != sub_24853D0(*(_QWORD *)v7, *(_DWORD *)(v7 + 16), *(_DWORD *)(v7 + 20)) )
      break;
    v7 += 32;
    ++a3;
    if ( a1 + v4 == v7 )
      return 1;
  }
  return 0;
}
