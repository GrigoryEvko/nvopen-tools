// Function: sub_27D83B0
// Address: 0x27d83b0
//
__int64 __fastcall sub_27D83B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // r15
  __int64 i; // r13
  char *v6; // rsi
  int v7; // eax
  __int64 k; // r14
  char *v9; // rsi
  int v10; // eax
  __int64 j; // [rsp+10h] [rbp-70h]
  __int64 v13; // [rsp+18h] [rbp-68h]
  _QWORD v16[10]; // [rsp+30h] [rbp-50h] BYREF

  v3 = 0;
  v4 = sub_B2BEC0(a1);
  v13 = *(_QWORD *)(a1 + 80);
  if ( v13 != a1 + 72 )
  {
    do
    {
      if ( !v13 )
        BUG();
      for ( i = *(_QWORD *)(v13 + 32); v13 + 24 != i; v3 |= v7 )
      {
        v6 = (char *)(i - 24);
        if ( !i )
          v6 = 0;
        v16[0] = v4;
        v7 = sub_27D81A0(v4, v6, (__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))sub_27D8160, (__int64)v16);
        i = *(_QWORD *)(i + 8);
      }
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( a1 + 72 != v13 );
    for ( j = *(_QWORD *)(a1 + 80); v13 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      for ( k = *(_QWORD *)(j + 32); j + 24 != k; v3 |= v10 )
      {
        v9 = (char *)(k - 24);
        if ( !k )
          v9 = 0;
        v16[0] = v4;
        v16[1] = a2;
        v16[2] = v9;
        v16[3] = a3;
        v10 = sub_27D81A0(v4, v9, (__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))sub_27D82A0, (__int64)v16);
        k = *(_QWORD *)(k + 8);
      }
    }
  }
  return v3;
}
