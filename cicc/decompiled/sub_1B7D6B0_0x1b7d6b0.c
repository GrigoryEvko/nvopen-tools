// Function: sub_1B7D6B0
// Address: 0x1b7d6b0
//
__int64 __fastcall sub_1B7D6B0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r9
  __int64 v4; // rbx
  _QWORD *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r10d
  __int64 v11; // r11
  __int64 v12; // r8
  __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(*a1 + 40LL);
  v3 = *(_QWORD *)(v2 + 48);
  v4 = v2 + 40;
  if ( v3 == v4 )
    return *a1 + 24LL;
  v6 = &a1[a2];
  do
  {
    v7 = v3 - 24;
    if ( !v3 )
      v7 = 0;
    v14[0] = v7;
    if ( v6 != sub_1B7D2F0(a1, (__int64)v6, v14) )
    {
      v12 = v8 + 24;
      if ( !v10 )
        v11 = v12;
      if ( a2 == v10 + 1 )
        break;
    }
    v3 = *(_QWORD *)(v9 + 8);
  }
  while ( v4 != v3 );
  return v11;
}
