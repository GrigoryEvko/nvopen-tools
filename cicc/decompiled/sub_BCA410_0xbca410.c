// Function: sub_BCA410
// Address: 0xbca410
//
__int64 __fastcall sub_BCA410(__int64 a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int *v5; // r15
  _BYTE *v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  unsigned int *i; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v18; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v19; // [rsp+28h] [rbp-68h]
  _BYTE *v20; // [rsp+30h] [rbp-60h]
  __int64 v21; // [rsp+40h] [rbp-50h] BYREF
  __int64 v22; // [rsp+48h] [rbp-48h]
  _QWORD *v23; // [rsp+50h] [rbp-40h]

  v18 = 0;
  v19 = 0;
  v20 = 0;
  v3 = sub_BCB2D0(a2);
  v4 = sub_BCB2E0(a2);
  v5 = *(unsigned int **)(a1 + 8);
  for ( i = *(unsigned int **)(a1 + 16); i != v5; v19 = v6 + 8 )
  {
    while ( 1 )
    {
      v7 = *v5;
      v8 = sub_AD64C0(v3, v7, 0);
      v21 = (__int64)sub_B98A20(v8, v7);
      v9 = *((_QWORD *)v5 + 1);
      v10 = sub_AD64C0(v4, v9, 0);
      v22 = (__int64)sub_B98A20(v10, v9);
      v11 = *((_QWORD *)v5 + 2);
      v12 = sub_AD64C0(v3, v11, 0);
      v23 = sub_B98A20(v12, v11);
      v13 = sub_B9C770(a2, &v21, (__int64 *)3, 0, 1);
      v6 = v19;
      v17 = v13;
      if ( v19 != v20 )
        break;
      v5 += 6;
      sub_914280((__int64)&v18, v19, &v17);
      if ( i == v5 )
        goto LABEL_8;
    }
    if ( v19 )
    {
      *(_QWORD *)v19 = v13;
      v6 = v19;
    }
    v5 += 6;
  }
LABEL_8:
  v21 = sub_B9B140(a2, "DetailedSummary", 0xFu);
  v22 = sub_B9C770(a2, v18, (__int64 *)((v19 - (_BYTE *)v18) >> 3), 0, 1);
  v14 = sub_B9C770(a2, &v21, (__int64 *)2, 0, 1);
  if ( v18 )
    j_j___libc_free_0(v18, v20 - (_BYTE *)v18);
  return v14;
}
