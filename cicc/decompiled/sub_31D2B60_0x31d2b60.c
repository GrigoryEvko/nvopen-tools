// Function: sub_31D2B60
// Address: 0x31d2b60
//
__int64 __fastcall sub_31D2B60(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  char v4; // cl
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  char v7; // cl
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v12; // rbx
  __int64 v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rbx
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // rsi
  _QWORD *v23; // rbx
  _QWORD *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-128h] BYREF
  __int64 v27; // [rsp+10h] [rbp-120h]
  __int64 v28; // [rsp+18h] [rbp-118h]
  __int64 v29; // [rsp+20h] [rbp-110h]
  void *v30; // [rsp+30h] [rbp-100h]
  __int64 v31; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v32; // [rsp+40h] [rbp-F0h]
  __int64 v33; // [rsp+48h] [rbp-E8h]
  __int64 i; // [rsp+50h] [rbp-E0h]
  __int64 v35; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v36; // [rsp+68h] [rbp-C8h]
  __int64 v37; // [rsp+70h] [rbp-C0h]
  unsigned int v38; // [rsp+78h] [rbp-B8h]
  _QWORD *v39; // [rsp+88h] [rbp-A8h]
  unsigned int v40; // [rsp+98h] [rbp-98h]
  char v41; // [rsp+A0h] [rbp-90h]
  _QWORD v42[11]; // [rsp+A8h] [rbp-88h] BYREF

  v35 = 0;
  memset(v42, 0, sizeof(v42));
  v38 = 128;
  v2 = (_QWORD *)sub_C7D670(6144, 8);
  v37 = 0;
  v36 = v2;
  v31 = 2;
  v30 = &unk_4A259B8;
  v32 = 0;
  v3 = v2 + 768;
  v33 = -4096;
  for ( i = 0; v3 != v2; v2 += 6 )
  {
    if ( v2 )
    {
      v4 = v31;
      v2[2] = 0;
      v2[3] = -4096;
      *v2 = &unk_4A259B8;
      v2[1] = v4 & 6;
      v2[4] = i;
    }
  }
  v41 = 0;
  v42[1] = 0;
  LODWORD(v42[4]) = 128;
  v5 = (_QWORD *)sub_C7D670(6144, 8);
  v42[3] = 0;
  v42[2] = v5;
  v31 = 2;
  v30 = &unk_4A34DD0;
  v32 = 0;
  v6 = &v5[6 * LODWORD(v42[4])];
  v33 = -4096;
  for ( i = 0; v6 != v5; v5 += 6 )
  {
    if ( v5 )
    {
      v7 = v31;
      v5[2] = 0;
      v5[3] = -4096;
      *v5 = &unk_4A34DD0;
      v5[1] = v7 & 6;
      v5[4] = i;
    }
  }
  LOBYTE(v42[9]) = 0;
  v8 = sub_31D1050(&v35, a2);
  if ( LOBYTE(v42[9]) )
  {
    v18 = LODWORD(v42[8]);
    LOBYTE(v42[9]) = 0;
    if ( LODWORD(v42[8]) )
    {
      v19 = (_QWORD *)v42[6];
      v20 = v42[6] + 16LL * LODWORD(v42[8]);
      do
      {
        if ( *v19 != -4096 && *v19 != -8192 )
        {
          v21 = v19[1];
          if ( v21 )
            sub_B91220((__int64)(v19 + 1), v21);
        }
        v19 += 2;
      }
      while ( (_QWORD *)v20 != v19 );
      v18 = LODWORD(v42[8]);
    }
    sub_C7D6A0(v42[6], 16 * v18, 8);
  }
  v9 = LODWORD(v42[4]);
  if ( LODWORD(v42[4]) )
  {
    v12 = (_QWORD *)v42[2];
    v26 = 2;
    v13 = v42[2] + 48LL * LODWORD(v42[4]);
    v27 = 0;
    v28 = -4096;
    v29 = 0;
    v31 = 2;
    v32 = 0;
    v33 = -8192;
    v30 = &unk_4A34DD0;
    i = 0;
    do
    {
      v14 = v12[3];
      *v12 = &unk_49DB368;
      if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
        sub_BD60C0(v12 + 1);
      v12 += 6;
    }
    while ( (_QWORD *)v13 != v12 );
    v30 = &unk_49DB368;
    if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
      sub_BD60C0(&v31);
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
      sub_BD60C0(&v26);
    v9 = LODWORD(v42[4]);
  }
  sub_C7D6A0(v42[2], 48 * v9, 8);
  if ( v41 )
  {
    v22 = v40;
    v41 = 0;
    if ( v40 )
    {
      v23 = v39;
      v24 = &v39[2 * v40];
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
      v22 = v40;
    }
    sub_C7D6A0((__int64)v39, 16 * v22, 8);
  }
  v10 = v38;
  if ( v38 )
  {
    v15 = v36;
    v26 = 2;
    v27 = 0;
    v16 = &v36[6 * v38];
    v28 = -4096;
    v29 = 0;
    v31 = 2;
    v32 = 0;
    v33 = -8192;
    v30 = &unk_4A259B8;
    i = 0;
    do
    {
      v17 = v15[3];
      *v15 = &unk_49DB368;
      if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
        sub_BD60C0(v15 + 1);
      v15 += 6;
    }
    while ( v16 != v15 );
    v30 = &unk_49DB368;
    if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
      sub_BD60C0(&v31);
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
      sub_BD60C0(&v26);
    v10 = v38;
  }
  sub_C7D6A0((__int64)v36, 48 * v10, 8);
  return v8;
}
