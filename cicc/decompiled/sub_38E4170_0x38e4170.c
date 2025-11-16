// Function: sub_38E4170
// Address: 0x38e4170
//
__int64 __fastcall sub_38E4170(_QWORD *a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v6; // r15
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r13
  unsigned __int64 *v17; // r14
  bool v18; // bl
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // r13
  unsigned __int64 *v25; // r14
  bool v26; // bl
  __int64 *v27; // rdi
  unsigned __int64 v28; // [rsp+8h] [rbp-128h]
  unsigned __int64 v29; // [rsp+8h] [rbp-128h]
  __int16 v34; // [rsp+40h] [rbp-F0h]
  __int16 v35; // [rsp+40h] [rbp-F0h]
  _QWORD *v36; // [rsp+48h] [rbp-E8h] BYREF
  _QWORD v37[2]; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v38[2]; // [rsp+68h] [rbp-C8h] BYREF
  _QWORD v39[2]; // [rsp+78h] [rbp-B8h] BYREF
  __int64 *v40; // [rsp+88h] [rbp-A8h]
  __int64 *v41; // [rsp+90h] [rbp-A0h]
  char *v42; // [rsp+98h] [rbp-98h]
  unsigned __int64 v43; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD *v44; // [rsp+A8h] [rbp-88h] BYREF
  _QWORD v45[2]; // [rsp+B8h] [rbp-78h] BYREF
  __int64 v46[2]; // [rsp+C8h] [rbp-68h] BYREF
  _QWORD v47[2]; // [rsp+D8h] [rbp-58h] BYREF
  __int64 *v48; // [rsp+E8h] [rbp-48h]
  __int64 *v49; // [rsp+F0h] [rbp-40h]
  char *v50; // [rsp+F8h] [rbp-38h]

  v6 = a1[1];
  v7 = *(_BYTE **)(v6 + 56);
  v8 = *(_QWORD *)(v6 + 64);
  v35 = *(_WORD *)(v6 + 48) & 0x3FFF | v34 & 0xC000;
  v36 = v37;
  sub_38E3500((__int64 *)&v36, v7, (__int64)&v7[v8]);
  v9 = *(_BYTE **)(v6 + 88);
  v10 = *(_QWORD *)(v6 + 96);
  v11 = v38;
  v38[0] = (__int64)v39;
  sub_38E3500(v38, v9, (__int64)&v9[v10]);
  v12 = *(_QWORD *)(v6 + 128);
  v13 = *(_QWORD *)(v6 + 120);
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v14 = v12 - v13;
  if ( v12 == v13 )
  {
    v16 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_46;
    v28 = v12 - v13;
    v15 = sub_22077B0(v12 - v13);
    v12 = *(_QWORD *)(v6 + 128);
    v13 = *(_QWORD *)(v6 + 120);
    v14 = v28;
    v16 = (__int64 *)v15;
  }
  v40 = v16;
  v41 = v16;
  v42 = (char *)v16 + v14;
  if ( v13 == v12 )
  {
    v18 = (v35 & 0x10) != 0;
    goto LABEL_13;
  }
  do
  {
    if ( v16 )
    {
      *v16 = (__int64)(v16 + 2);
      sub_38E3500(v16, *(_BYTE **)v13, *(_QWORD *)v13 + *(_QWORD *)(v13 + 8));
    }
    v13 += 32;
    v16 += 4;
  }
  while ( v12 != v13 );
  v17 = (unsigned __int64 *)v40;
  v41 = v16;
  v18 = (v35 & 0x10) != 0;
  if ( v16 != v40 )
  {
    do
    {
      if ( (unsigned __int64 *)*v17 != v17 + 2 )
        j_j___libc_free_0(*v17);
      v17 += 4;
    }
    while ( v16 != (__int64 *)v17 );
    v16 = v40;
LABEL_13:
    if ( !v16 )
      goto LABEL_15;
  }
  j_j___libc_free_0((unsigned __int64)v16);
LABEL_15:
  if ( (_QWORD *)v38[0] != v39 )
    j_j___libc_free_0(v38[0]);
  if ( v36 != v37 )
    j_j___libc_free_0((unsigned __int64)v36);
  if ( v18 )
    return 0;
  v20 = a1[1];
  LOWORD(v43) = *(_WORD *)(v20 + 48) & 0x3FFF | v43 & 0xC000;
  HIDWORD(v43) = *(_DWORD *)(v20 + 52);
  v44 = v45;
  sub_38E3500((__int64 *)&v44, *(_BYTE **)(v20 + 56), *(_QWORD *)(v20 + 56) + *(_QWORD *)(v20 + 64));
  v11 = v46;
  v46[0] = (__int64)v47;
  v9 = *(_BYTE **)(v20 + 88);
  sub_38E3500(v46, v9, (__int64)&v9[*(_QWORD *)(v20 + 96)]);
  v21 = *(_QWORD *)(v20 + 128);
  v22 = *(_QWORD *)(v20 + 120);
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v14 = v21 - v22;
  if ( v21 != v22 )
  {
    if ( v14 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v29 = v21 - v22;
      v23 = sub_22077B0(v21 - v22);
      v21 = *(_QWORD *)(v20 + 128);
      v22 = *(_QWORD *)(v20 + 120);
      v14 = v29;
      v24 = (__int64 *)v23;
      goto LABEL_24;
    }
LABEL_46:
    sub_4261EA(v11, v9, v14);
  }
  v24 = 0;
LABEL_24:
  v48 = v24;
  v49 = v24;
  v50 = (char *)v24 + v14;
  if ( v21 == v22 )
  {
    v26 = (v43 & 8) != 0;
    goto LABEL_33;
  }
  do
  {
    if ( v24 )
    {
      *v24 = (__int64)(v24 + 2);
      sub_38E3500(v24, *(_BYTE **)v22, *(_QWORD *)v22 + *(_QWORD *)(v22 + 8));
    }
    v22 += 32;
    v24 += 4;
  }
  while ( v21 != v22 );
  v25 = (unsigned __int64 *)v48;
  v49 = v24;
  v26 = (v43 & 8) != 0;
  if ( v48 != v24 )
  {
    do
    {
      if ( (unsigned __int64 *)*v25 != v25 + 2 )
        j_j___libc_free_0(*v25);
      v25 += 4;
    }
    while ( v24 != (__int64 *)v25 );
    v24 = v48;
LABEL_33:
    if ( !v24 )
      goto LABEL_35;
  }
  j_j___libc_free_0((unsigned __int64)v24);
LABEL_35:
  if ( (_QWORD *)v46[0] != v47 )
    j_j___libc_free_0(v46[0]);
  if ( v44 != v45 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( !v26 )
  {
    v27 = (__int64 *)a1[43];
    v43 = a4;
    v44 = (_QWORD *)a5;
    sub_16D14E0(v27, a2, 1, a3, &v43, 1, 0, 0, 1u);
    sub_38E35B0(a1);
    return 0;
  }
  return sub_3909790(a1, a2, a3, a4, a5);
}
