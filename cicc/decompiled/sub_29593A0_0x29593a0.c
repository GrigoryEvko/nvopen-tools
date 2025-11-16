// Function: sub_29593A0
// Address: 0x29593a0
//
__int64 __fastcall sub_29593A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  bool v10; // al
  __int64 *v11; // r8
  __int64 v12; // r13
  bool v13; // al
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 *v20; // [rsp+28h] [rbp-38h]
  __int64 v21; // [rsp+28h] [rbp-38h]

  v18 = (a3 - 1) / 2;
  v17 = a3 & 1;
  if ( a2 >= v18 )
  {
    v7 = a2;
    v11 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = 16 * (i + 1);
    v9 = *(_QWORD *)(a1 + v8);
    v20 = (__int64 *)(a1 + v8);
    v10 = sub_2959010(a5, v9, *(_QWORD *)(a1 + v8 - 8));
    v11 = v20;
    if ( v10 )
    {
      --v7;
      v11 = (__int64 *)(a1 + 8 * v7);
      v9 = *v11;
    }
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v7 >= v18 )
      break;
  }
  if ( !v17 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v7 )
    {
      v15 = *(_QWORD *)(a1 + 8 * (2 * v7 + 2) - 8);
      v7 = 2 * v7 + 1;
      *v11 = v15;
      v11 = (__int64 *)(a1 + 8 * v7);
    }
  }
  v12 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(a1 + 8 * v12);
      v13 = sub_2959010(a5, v21, a4);
      v11 = (__int64 *)(a1 + 8 * v7);
      if ( !v13 )
        break;
      v7 = v12;
      *v11 = v21;
      if ( a2 >= v12 )
      {
        v11 = (__int64 *)(a1 + 8 * v12);
        break;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v11 = a4;
  return a4;
}
