// Function: sub_31D30F0
// Address: 0x31d30f0
//
_QWORD *__fastcall sub_31D30F0(_QWORD *a1, __int64 a2, __int64 *a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  char v6; // cl
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  char v9; // cl
  char v10; // al
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r13
  __int64 v21; // rax
  __int64 v22; // rsi
  _QWORD *v23; // rbx
  _QWORD *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rsi
  _QWORD *v27; // rbx
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // [rsp+8h] [rbp-128h] BYREF
  __int64 v31; // [rsp+10h] [rbp-120h]
  __int64 v32; // [rsp+18h] [rbp-118h]
  __int64 v33; // [rsp+20h] [rbp-110h]
  void *v34; // [rsp+30h] [rbp-100h]
  __int64 v35; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v36; // [rsp+40h] [rbp-F0h]
  __int64 v37; // [rsp+48h] [rbp-E8h]
  __int64 i; // [rsp+50h] [rbp-E0h]
  __int64 v39; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v40; // [rsp+68h] [rbp-C8h]
  __int64 v41; // [rsp+70h] [rbp-C0h]
  unsigned int v42; // [rsp+78h] [rbp-B8h]
  _QWORD *v43; // [rsp+88h] [rbp-A8h]
  unsigned int v44; // [rsp+98h] [rbp-98h]
  char v45; // [rsp+A0h] [rbp-90h]
  _QWORD v46[11]; // [rsp+A8h] [rbp-88h] BYREF

  v39 = 0;
  memset(v46, 0, sizeof(v46));
  v42 = 128;
  v4 = (_QWORD *)sub_C7D670(6144, 8);
  v41 = 0;
  v40 = v4;
  v35 = 2;
  v34 = &unk_4A259B8;
  v36 = 0;
  v5 = v4 + 768;
  v37 = -4096;
  for ( i = 0; v5 != v4; v4 += 6 )
  {
    if ( v4 )
    {
      v6 = v35;
      v4[2] = 0;
      v4[3] = -4096;
      *v4 = &unk_4A259B8;
      v4[1] = v6 & 6;
      v4[4] = i;
    }
  }
  v45 = 0;
  v46[1] = 0;
  LODWORD(v46[4]) = 128;
  v7 = (_QWORD *)sub_C7D670(6144, 8);
  v46[3] = 0;
  v46[2] = v7;
  v35 = 2;
  v34 = &unk_4A34DD0;
  v36 = 0;
  v8 = &v7[6 * LODWORD(v46[4])];
  v37 = -4096;
  for ( i = 0; v8 != v7; v7 += 6 )
  {
    if ( v7 )
    {
      v9 = v35;
      v7[2] = 0;
      v7[3] = -4096;
      *v7 = &unk_4A34DD0;
      v7[1] = v9 & 6;
      v7[4] = i;
    }
  }
  LOBYTE(v46[9]) = 0;
  v10 = sub_31D1050(&v39, a3);
  v11 = a1 + 4;
  v12 = a1 + 10;
  if ( v10 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v11;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v12;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v11;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v12;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  if ( LOBYTE(v46[9]) )
  {
    v26 = LODWORD(v46[8]);
    LOBYTE(v46[9]) = 0;
    if ( LODWORD(v46[8]) )
    {
      v27 = (_QWORD *)v46[6];
      v28 = v46[6] + 16LL * LODWORD(v46[8]);
      do
      {
        if ( *v27 != -4096 && *v27 != -8192 )
        {
          v29 = v27[1];
          if ( v29 )
            sub_B91220((__int64)(v27 + 1), v29);
        }
        v27 += 2;
      }
      while ( (_QWORD *)v28 != v27 );
      v26 = LODWORD(v46[8]);
    }
    sub_C7D6A0(v46[6], 16 * v26, 8);
  }
  v13 = LODWORD(v46[4]);
  if ( LODWORD(v46[4]) )
  {
    v16 = (_QWORD *)v46[2];
    v30 = 2;
    v17 = v46[2] + 48LL * LODWORD(v46[4]);
    v31 = 0;
    v32 = -4096;
    v33 = 0;
    v35 = 2;
    v36 = 0;
    v37 = -8192;
    v34 = &unk_4A34DD0;
    i = 0;
    do
    {
      v18 = v16[3];
      *v16 = &unk_49DB368;
      if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
        sub_BD60C0(v16 + 1);
      v16 += 6;
    }
    while ( (_QWORD *)v17 != v16 );
    v34 = &unk_49DB368;
    if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
      sub_BD60C0(&v35);
    if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
      sub_BD60C0(&v30);
    v13 = LODWORD(v46[4]);
  }
  sub_C7D6A0(v46[2], 48 * v13, 8);
  if ( v45 )
  {
    v22 = v44;
    v45 = 0;
    if ( v44 )
    {
      v23 = v43;
      v24 = &v43[2 * v44];
      do
      {
        if ( *v23 != -4096 && *v23 != -8192 )
        {
          v25 = v23[1];
          if ( v25 )
            sub_B91220((__int64)(v23 + 1), v25);
        }
        v23 += 2;
      }
      while ( v24 != v23 );
      v22 = v44;
    }
    sub_C7D6A0((__int64)v43, 16 * v22, 8);
  }
  v14 = v42;
  if ( v42 )
  {
    v19 = v40;
    v30 = 2;
    v31 = 0;
    v20 = &v40[6 * v42];
    v32 = -4096;
    v33 = 0;
    v35 = 2;
    v36 = 0;
    v37 = -8192;
    v34 = &unk_4A259B8;
    i = 0;
    do
    {
      v21 = v19[3];
      *v19 = &unk_49DB368;
      if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
        sub_BD60C0(v19 + 1);
      v19 += 6;
    }
    while ( v20 != v19 );
    v34 = &unk_49DB368;
    if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
      sub_BD60C0(&v35);
    if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
      sub_BD60C0(&v30);
    v14 = v42;
  }
  sub_C7D6A0((__int64)v40, 48 * v14, 8);
  return a1;
}
