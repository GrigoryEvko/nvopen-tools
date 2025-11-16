// Function: sub_2CFC5B0
// Address: 0x2cfc5b0
//
__int64 __fastcall sub_2CFC5B0(__int64 a1, __int64 a2, __int64 a3)
{
  void **v4; // rax
  void **v5; // rdx
  char v6; // cl
  char v7; // r12
  __int64 v8; // rax
  void *v9; // rsi
  void **v11; // rbx
  void **v12; // r14
  char *v13; // rax
  unsigned int v14; // eax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  __int64 v17; // rsi
  _QWORD v18[2]; // [rsp+8h] [rbp-E8h] BYREF
  __int64 v19; // [rsp+18h] [rbp-D8h]
  __int64 v20; // [rsp+20h] [rbp-D0h]
  void *v21; // [rsp+30h] [rbp-C0h]
  __int64 v22; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v23; // [rsp+40h] [rbp-B0h]
  __int64 v24; // [rsp+48h] [rbp-A8h]
  void *i; // [rsp+50h] [rbp-A0h]
  __int64 v26; // [rsp+60h] [rbp-90h] BYREF
  void **v27; // [rsp+68h] [rbp-88h]
  __int64 v28; // [rsp+70h] [rbp-80h]
  unsigned int v29; // [rsp+78h] [rbp-78h]
  char v30; // [rsp+7Ch] [rbp-74h]
  void *v31; // [rsp+80h] [rbp-70h] BYREF
  _QWORD *v32; // [rsp+88h] [rbp-68h]
  __int64 v33; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v34; // [rsp+98h] [rbp-58h]
  __int64 v35; // [rsp+A0h] [rbp-50h]
  _BYTE v36[5]; // [rsp+A8h] [rbp-48h]
  __int16 v37; // [rsp+ADh] [rbp-43h]
  char v38; // [rsp+AFh] [rbp-41h]
  _BYTE v39[64]; // [rsp+B0h] [rbp-40h] BYREF

  *(_DWORD *)&v36[1] = 0;
  v37 = 0;
  v38 = 0;
  v26 = 0;
  v29 = 128;
  v4 = (void **)sub_C7D670(6144, 8);
  v28 = 0;
  v27 = v4;
  v22 = 2;
  v21 = &unk_4A259B8;
  v23 = 0;
  v5 = v4 + 768;
  v24 = -4096;
  for ( i = 0; v5 != v4; v4 += 6 )
  {
    if ( v4 )
    {
      v6 = v22;
      v4[2] = 0;
      v4[3] = (void *)-4096LL;
      *v4 = &unk_4A259B8;
      v4[1] = (void *)(v6 & 6);
      v4[4] = i;
    }
  }
  v7 = qword_5014888;
  LOBYTE(v35) = 0;
  if ( (_BYTE)qword_5014888 )
  {
    v7 = sub_2CFAA90((__int64)&v26, a3);
    if ( (_BYTE)v35 )
    {
      v14 = v34;
      LOBYTE(v35) = 0;
      if ( (_DWORD)v34 )
      {
        v15 = v32;
        v16 = &v32[2 * (unsigned int)v34];
        do
        {
          if ( *v15 != -8192 && *v15 != -4096 )
          {
            v17 = v15[1];
            if ( v17 )
              sub_B91220((__int64)(v15 + 1), v17);
          }
          v15 += 2;
        }
        while ( v16 != v15 );
        v14 = v34;
      }
      sub_C7D6A0((__int64)v32, 16LL * v14, 8);
    }
  }
  v8 = v29;
  if ( v29 )
  {
    v11 = v27;
    v18[0] = 2;
    v12 = &v27[6 * v29];
    v18[1] = 0;
    v19 = -4096;
    v20 = 0;
    v22 = 2;
    v23 = 0;
    v24 = -8192;
    v21 = &unk_4A259B8;
    i = 0;
    do
    {
      v13 = (char *)v11[3];
      *v11 = &unk_49DB368;
      if ( v13 != 0 && v13 + 4096 != 0 && v13 != (char *)-8192LL )
        sub_BD60C0(v11 + 1);
      v11 += 6;
    }
    while ( v12 != v11 );
    v21 = &unk_49DB368;
    if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
      sub_BD60C0(&v22);
    if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
      sub_BD60C0(v18);
    v8 = v29;
  }
  sub_C7D6A0((__int64)v27, 48 * v8, 8);
  v9 = (void *)(a1 + 32);
  if ( v7 )
  {
    v27 = &v31;
    v28 = 0x100000002LL;
    v31 = &unk_4F82408;
    v29 = 0;
    v30 = 1;
    v33 = 0;
    v34 = (unsigned __int64)v39;
    v35 = 2;
    *(_DWORD *)v36 = 0;
    v36[4] = 1;
    v26 = 1;
    sub_C8CF70(a1, v9, 2, (__int64)&v31, (__int64)&v26);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v39, (__int64)&v33);
    if ( v36[4] )
    {
      if ( v30 )
        return a1;
    }
    else
    {
      _libc_free(v34);
      if ( v30 )
        return a1;
    }
    _libc_free((unsigned __int64)v27);
    return a1;
  }
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  return a1;
}
