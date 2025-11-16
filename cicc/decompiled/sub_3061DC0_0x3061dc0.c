// Function: sub_3061DC0
// Address: 0x3061dc0
//
void __fastcall sub_3061DC0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // r14
  char v5; // bl
  __int64 v6; // rax
  __int64 v7; // rdi
  char *v8; // rsi
  char v9; // bl
  char v10; // r14
  __int64 v11; // rax
  char v12; // bl
  char v13; // r14
  __int64 v14; // rdi
  char *v15; // rsi
  __int64 v16; // r13
  char *v17; // r12
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-68h] BYREF
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  char v22; // [rsp+18h] [rbp-58h]
  char *v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]
  char *v25; // [rsp+30h] [rbp-40h]
  __int64 v26; // [rsp+38h] [rbp-38h]
  __int64 v27; // [rsp+40h] [rbp-30h]

  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  if ( !(_BYTE)qword_502BBC8 )
  {
    nullsub_1908(&v20);
    v18 = sub_22077B0(0x10u);
    if ( v18 )
      *(_QWORD *)v18 = &unk_4A31060;
    v21 = v18;
    if ( (char *)v24 == v25 )
    {
      sub_2353750((unsigned __int64 *)&v23, (char *)v24, &v21);
      v18 = v21;
    }
    else
    {
      if ( v24 )
      {
        *(_QWORD *)v24 = v18;
        v24 += 8;
        goto LABEL_2;
      }
      v24 = 8;
    }
    if ( v18 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
  }
LABEL_2:
  v3 = sub_22077B0(0x10u);
  if ( v3 )
    *(_QWORD *)v3 = &unk_4A30FE0;
  v21 = v3;
  if ( (char *)v24 == v25 )
  {
    sub_2353750((unsigned __int64 *)&v23, (char *)v24, &v21);
    v3 = v21;
  }
  else
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = v3;
      v24 += 8;
      goto LABEL_7;
    }
    v24 = 8;
  }
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
LABEL_7:
  if ( !(_BYTE)qword_502B748 )
    goto LABEL_8;
  v19 = sub_22077B0(0x10u);
  if ( v19 )
    *(_QWORD *)v19 = &unk_4A310A0;
  v21 = v19;
  if ( (char *)v24 == v25 )
  {
    sub_2353750((unsigned __int64 *)&v23, (char *)v24, &v21);
    v19 = v21;
  }
  else
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = v19;
      v24 += 8;
      goto LABEL_8;
    }
    v24 = 8;
  }
  if ( v19 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
LABEL_8:
  sub_234AAB0((__int64)&v21, (__int64 *)&v23, 0);
  v4 = v21;
  v5 = v22;
  v21 = 0;
  v6 = sub_22077B0(0x18u);
  v7 = v6;
  if ( v6 )
  {
    *(_BYTE *)(v6 + 16) = v5;
    *(_QWORD *)(v6 + 8) = v4;
    v4 = 0;
    *(_QWORD *)v6 = &unk_4A0C478;
  }
  v20 = v6;
  v8 = (char *)a2[1];
  if ( v8 == (char *)a2[2] )
  {
    sub_2275C60(a2, v8, &v20);
    v7 = v20;
  }
  else
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = v6;
      a2[1] += 8LL;
      goto LABEL_13;
    }
    a2[1] = 8;
  }
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
LABEL_13:
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  if ( v21 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
  v9 = byte_502BD88;
  v10 = byte_502BCA8;
  v11 = sub_22077B0(0x10u);
  v12 = v9 ^ 1;
  v13 = v10 ^ 1;
  v14 = v11;
  if ( v11 )
  {
    *(_BYTE *)(v11 + 8) = v12;
    *(_BYTE *)(v11 + 9) = v13;
    *(_QWORD *)v11 = &unk_4A30F60;
  }
  v21 = v11;
  v15 = (char *)a2[1];
  if ( v15 == (char *)a2[2] )
  {
    sub_2275C60(a2, v15, &v21);
    v14 = v21;
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v11;
      a2[1] += 8LL;
      goto LABEL_22;
    }
    a2[1] = 8;
  }
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
LABEL_22:
  v16 = v24;
  v17 = v23;
  if ( (char *)v24 != v23 )
  {
    do
    {
      if ( *(_QWORD *)v17 )
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v17 + 8LL))(*(_QWORD *)v17);
      v17 += 8;
    }
    while ( (char *)v16 != v17 );
    v17 = v23;
  }
  if ( v17 )
    j_j___libc_free_0((unsigned __int64)v17);
}
