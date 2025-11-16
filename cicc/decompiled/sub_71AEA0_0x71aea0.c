// Function: sub_71AEA0
// Address: 0x71aea0
//
__int64 __fastcall sub_71AEA0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 i; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 j; // r15
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 *v15; // [rsp+8h] [rbp-38h]

  for ( i = *(_QWORD *)(a3 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v15 = *(__int64 **)(i + 168);
  v8 = sub_6963A0(a1, *v15, a4);
  v9 = *a2;
  for ( j = v8; *(_BYTE *)(v9 + 140) == 12; v9 = *(_QWORD *)(v9 + 160) )
    ;
  v11 = v15[5];
  v12 = sub_73E1B0(a2, v11);
  if ( v11 != v9 )
  {
    v13 = sub_8D5CE0(v9, v11);
    if ( v13 )
      v12 = sub_73E4A0(v12, v13);
  }
  return sub_701AC0(a3, 1, v12, j, a4);
}
