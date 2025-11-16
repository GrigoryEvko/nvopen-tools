// Function: sub_2C16E90
// Address: 0x2c16e90
//
__int64 __fastcall sub_2C16E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  const void *v8; // r14
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 *v12; // rdi
  __int64 *v14; // rdi
  char v15; // [rsp+Fh] [rbp-B1h]
  __int64 v16; // [rsp+10h] [rbp-B0h]
  __int64 *v17; // [rsp+18h] [rbp-A8h]
  __int64 v18; // [rsp+28h] [rbp-98h] BYREF
  __int64 v19; // [rsp+30h] [rbp-90h] BYREF
  __int64 v20; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v21; // [rsp+40h] [rbp-80h] BYREF
  __int64 v22; // [rsp+48h] [rbp-78h]
  _BYTE dest[16]; // [rsp+50h] [rbp-70h] BYREF
  void *v24; // [rsp+60h] [rbp-60h] BYREF
  __int16 v25; // [rsp+80h] [rbp-40h]

  v7 = *(unsigned int *)(a1 + 56);
  v21 = (__int64 *)dest;
  v22 = 0x200000000LL;
  v8 = *(const void **)(a1 + 48);
  v9 = 8 * v7;
  if ( v7 > 2 )
  {
    sub_C8D5F0((__int64)&v21, dest, v7, 8u, a5, a6);
    v14 = &v21[(unsigned int)v22];
  }
  else
  {
    v17 = (__int64 *)dest;
    if ( !v9 )
      goto LABEL_3;
    v14 = (__int64 *)dest;
  }
  memcpy(v14, v8, 8 * v7);
  LODWORD(v9) = v22;
  v17 = v21;
LABEL_3:
  LODWORD(v22) = v9 + v7;
  v16 = (unsigned int)(v9 + v7);
  v18 = *(_QWORD *)(a1 + 88);
  if ( v18 )
    sub_2AAAFA0(&v18);
  v25 = 260;
  v24 = (void *)(a1 + 168);
  v11 = sub_22077B0(0xC8u);
  if ( v11 )
  {
    v15 = *(_BYTE *)(a1 + 160);
    v19 = v18;
    if ( v18 )
    {
      sub_2AAAFA0(&v19);
      v20 = v19;
      if ( v19 )
        sub_2AAAFA0(&v20);
    }
    else
    {
      v20 = 0;
    }
    sub_2AAF4A0(v11, 4, v17, v16, &v20, v10);
    sub_9C6650(&v20);
    *(_BYTE *)(v11 + 152) = 7;
    *(_DWORD *)(v11 + 156) = 0;
    *(_QWORD *)v11 = &unk_4A23258;
    *(_QWORD *)(v11 + 40) = &unk_4A23290;
    *(_QWORD *)(v11 + 96) = &unk_4A232C8;
    sub_9C6650(&v19);
    *(_QWORD *)v11 = &unk_4A23B70;
    *(_QWORD *)(v11 + 96) = &unk_4A23BF0;
    *(_QWORD *)(v11 + 40) = &unk_4A23BB8;
    *(_BYTE *)(v11 + 160) = v15;
    sub_CA0F50((__int64 *)(v11 + 168), &v24);
  }
  sub_9C6650(&v18);
  v12 = v21;
  *(_BYTE *)(v11 + 152) = *(_BYTE *)(a1 + 152);
  *(_DWORD *)(v11 + 156) = *(_DWORD *)(a1 + 156);
  if ( v12 != (__int64 *)dest )
    _libc_free((unsigned __int64)v12);
  return v11;
}
