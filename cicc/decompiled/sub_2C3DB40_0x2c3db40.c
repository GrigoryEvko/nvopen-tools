// Function: sub_2C3DB40
// Address: 0x2c3db40
//
__int64 __fastcall sub_2C3DB40(_QWORD *a1, char a2, __int64 *a3, __int64 a4, int a5, char a6, __int64 *a7, void **a8)
{
  __int64 v9; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v17; // r9
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v22; // [rsp+38h] [rbp-58h] BYREF
  __int64 v23; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h] BYREF
  __int64 v25; // [rsp+50h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+58h] [rbp-38h] BYREF

  v9 = *a7;
  if ( a6 )
  {
    v26[0] = *a7;
    if ( v9 )
      sub_2AAAFA0(v26);
    v11 = sub_22077B0(0xC8u);
    v12 = v11;
    if ( v11 )
      sub_2C1AF80(v11, a2, a3, a4, a5, v26, a8);
    if ( *a1 )
    {
      v13 = (__int64 *)a1[1];
      *(_QWORD *)(v12 + 80) = *a1;
      v14 = *(_QWORD *)(v12 + 24);
      v15 = *v13;
      *(_QWORD *)(v12 + 32) = v13;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v12 + 24) = v15 | v14 & 7;
      *(_QWORD *)(v15 + 8) = v12 + 24;
      *v13 = *v13 & 7 | (v12 + 24);
    }
    sub_9C6650(v26);
    return v12;
  }
  v22 = *a7;
  if ( !v9 )
  {
    v23 = 0;
LABEL_25:
    v24 = 0;
    if ( !(8 * a4) )
      a3 = 0;
    goto LABEL_16;
  }
  sub_2AAAFA0(&v22);
  v23 = v22;
  if ( !v22 )
    goto LABEL_25;
  sub_2AAAFA0(&v23);
  v24 = v23;
  if ( !(8 * a4) )
    a3 = 0;
  if ( v23 )
    sub_2AAAFA0(&v24);
LABEL_16:
  v12 = sub_22077B0(0xC8u);
  if ( v12 )
  {
    v25 = v24;
    if ( v24 )
    {
      sub_2AAAFA0(&v25);
      v26[0] = v25;
      if ( v25 )
        sub_2AAAFA0(v26);
    }
    else
    {
      v26[0] = 0;
    }
    sub_2AAF4A0(v12, 4, a3, a4, v26, v17);
    sub_9C6650(v26);
    *(_BYTE *)(v12 + 152) = 7;
    *(_DWORD *)(v12 + 156) = 0;
    *(_QWORD *)v12 = &unk_4A23258;
    *(_QWORD *)(v12 + 40) = &unk_4A23290;
    *(_QWORD *)(v12 + 96) = &unk_4A232C8;
    sub_9C6650(&v25);
    *(_QWORD *)v12 = &unk_4A23B70;
    *(_QWORD *)(v12 + 96) = &unk_4A23BF0;
    *(_QWORD *)(v12 + 40) = &unk_4A23BB8;
    *(_BYTE *)(v12 + 160) = a2;
    sub_CA0F50((__int64 *)(v12 + 168), a8);
  }
  if ( *a1 )
  {
    v18 = (__int64 *)a1[1];
    *(_QWORD *)(v12 + 80) = *a1;
    v19 = *(_QWORD *)(v12 + 24);
    v20 = *v18;
    *(_QWORD *)(v12 + 32) = v18;
    v20 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v12 + 24) = v20 | v19 & 7;
    *(_QWORD *)(v20 + 8) = v12 + 24;
    *v18 = *v18 & 7 | (v12 + 24);
  }
  sub_9C6650(&v24);
  sub_9C6650(&v23);
  sub_9C6650(&v22);
  return v12;
}
