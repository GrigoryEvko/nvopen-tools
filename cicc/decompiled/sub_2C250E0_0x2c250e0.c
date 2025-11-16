// Function: sub_2C250E0
// Address: 0x2c250e0
//
__int64 __fastcall sub_2C250E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  __int64 i; // r15
  __int64 v8; // r12
  char v9; // al
  __int64 *v10; // r8
  __int64 v11; // r13
  __int64 *v12; // r15
  char v13; // al
  __int64 v15; // rax
  __int64 v17; // [rsp+10h] [rbp-50h]

  v6 = (a3 - 1) / 2;
  v17 = a3 & 1;
  if ( a2 >= v6 )
  {
    v8 = a2;
    v10 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = sub_2BFCAA0(a5, *(_QWORD *)(a1 + 16 * (i + 1)), *(_QWORD *)(a1 + 16 * (i + 1) - 8));
    v10 = (__int64 *)(a1 + 16 * (i + 1));
    if ( v9 )
    {
      --v8;
      v10 = (__int64 *)(a1 + 8 * v8);
    }
    *(_QWORD *)(a1 + 8 * i) = *v10;
    if ( v8 >= v6 )
      break;
  }
  if ( !v17 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v8 )
    {
      v15 = *(_QWORD *)(a1 + 8 * (2 * v8 + 2) - 8);
      v8 = 2 * v8 + 1;
      *v10 = v15;
      v10 = (__int64 *)(a1 + 8 * v8);
    }
  }
  v11 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 8 * v11);
      v13 = sub_2BFCAA0(a5, *v12, a4);
      v10 = (__int64 *)(a1 + 8 * v8);
      if ( !v13 )
        break;
      v8 = v11;
      *v10 = *v12;
      if ( a2 >= v11 )
      {
        v10 = (__int64 *)(a1 + 8 * v11);
        break;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = a4;
  return a4;
}
