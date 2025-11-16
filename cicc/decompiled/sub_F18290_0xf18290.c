// Function: sub_F18290
// Address: 0xf18290
//
_QWORD *__fastcall sub_F18290(const __m128i *a1, unsigned __int8 *a2)
{
  _BYTE *v2; // rdx
  char v3; // al
  _QWORD *v4; // r13
  _BYTE *v7; // r14
  unsigned __int64 v8; // rdx
  char v9; // al
  char *v10; // r15
  __int64 *v11; // r13
  unsigned int v12; // edx
  unsigned int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 v17; // rdi
  const void **v18; // rsi
  char **v19; // r13
  char **v20; // rax
  char *v21; // rdi
  char *v22; // rdi
  unsigned __int8 *v23; // rsi
  char *v24; // r14
  char *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  unsigned __int8 *v29; // [rsp-8h] [rbp-118h]
  char **v30; // [rsp+8h] [rbp-108h]
  char **v31; // [rsp+8h] [rbp-108h]
  unsigned __int64 v32; // [rsp+18h] [rbp-F8h]
  __int64 v34; // [rsp+28h] [rbp-E8h]
  _QWORD v35[2]; // [rsp+30h] [rbp-E0h] BYREF
  int v36; // [rsp+40h] [rbp-D0h]
  __int64 v37; // [rsp+48h] [rbp-C8h] BYREF
  int v38; // [rsp+50h] [rbp-C0h]
  unsigned __int64 v39; // [rsp+58h] [rbp-B8h]
  __int64 v40; // [rsp+60h] [rbp-B0h]
  int v41; // [rsp+68h] [rbp-A8h]
  __int64 v42; // [rsp+70h] [rbp-A0h]
  int v43; // [rsp+78h] [rbp-98h]
  char *v44; // [rsp+80h] [rbp-90h] BYREF
  __int64 v45; // [rsp+88h] [rbp-88h]
  char v46; // [rsp+90h] [rbp-80h] BYREF
  _BYTE v47[120]; // [rsp+98h] [rbp-78h] BYREF

  v2 = (_BYTE *)*((_QWORD *)a2 - 8);
  v3 = *v2;
  if ( *v2 <= 0x1Cu || v3 != 73 && v3 != 72 )
    return 0;
  v34 = *((_QWORD *)v2 - 4);
  if ( !v34 )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)a2 - 4);
  v8 = 0;
  v32 = 0;
  v9 = *v7;
  if ( *v7 > 0x15u )
  {
    if ( v9 == 73 || v9 == 72 )
    {
      v28 = *((_QWORD *)v7 - 4);
      v32 = v28;
      if ( v28 )
      {
        v7 = 0;
        v8 = v28 & 0xFFFFFFFFFFFFFFFBLL;
        goto LABEL_8;
      }
    }
    return 0;
  }
LABEL_8:
  v10 = &v46;
  v39 = v8;
  v11 = &v37;
  v36 = 1;
  v12 = 1;
  v35[1] = 0;
  v35[0] = v34 & 0xFFFFFFFFFFFFFFFBLL;
  v38 = 1;
  v37 = 0;
  v41 = 1;
  v40 = 0;
  v43 = 1;
  v42 = 0;
  v44 = &v46;
  v45 = 0x200000000LL;
  while ( 1 )
  {
    v15 = *(v11 - 3);
    *((_DWORD *)v10 + 4) = v12;
    *(_QWORD *)v10 = v15;
    if ( v12 > 0x40 )
      break;
    *((_QWORD *)v10 + 1) = *(v11 - 2);
    v13 = *((_DWORD *)v11 + 2);
    *((_DWORD *)v10 + 8) = v13;
    if ( v13 > 0x40 )
      goto LABEL_14;
LABEL_10:
    v14 = *v11;
    v11 += 5;
    v10 += 40;
    *((_QWORD *)v10 - 2) = v14;
    if ( v47 == (_BYTE *)v11 )
      goto LABEL_15;
LABEL_11:
    v12 = *((_DWORD *)v11 - 2);
  }
  sub_C43780((__int64)(v10 + 8), (const void **)v11 - 2);
  v16 = *((_DWORD *)v11 + 2);
  *((_DWORD *)v10 + 8) = v16;
  if ( v16 <= 0x40 )
    goto LABEL_10;
LABEL_14:
  v17 = (__int64)(v10 + 24);
  v18 = (const void **)v11;
  v11 += 5;
  v10 += 40;
  sub_C43780(v17, v18);
  if ( v47 != (_BYTE *)v11 )
    goto LABEL_11;
LABEL_15:
  LODWORD(v45) = v45 + 2;
  v19 = &v44;
  v20 = (char **)v35;
  do
  {
    v19 -= 5;
    if ( *((_DWORD *)v19 + 8) > 0x40u )
    {
      v21 = v19[3];
      if ( v21 )
      {
        v30 = v20;
        j_j___libc_free_0_0(v21);
        v20 = v30;
      }
    }
    if ( *((_DWORD *)v19 + 4) > 0x40u )
    {
      v22 = v19[1];
      if ( v22 )
      {
        v31 = v20;
        j_j___libc_free_0_0(v22);
        v20 = v31;
      }
    }
  }
  while ( v19 != v20 );
  v23 = v29;
  v4 = sub_F0E840(a1, a2, 0, v34, v32, (__int64)v7, (unsigned __int8 *)&v44);
  if ( !v4 )
  {
    v23 = a2;
    v4 = sub_F0E840(a1, a2, 1u, v34, v32, (__int64)v7, (unsigned __int8 *)&v44);
  }
  v24 = v44;
  v25 = &v44[40 * (unsigned int)v45];
  if ( v44 != v25 )
  {
    do
    {
      v25 -= 40;
      if ( *((_DWORD *)v25 + 8) > 0x40u )
      {
        v26 = *((_QWORD *)v25 + 3);
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
      if ( *((_DWORD *)v25 + 4) > 0x40u )
      {
        v27 = *((_QWORD *)v25 + 1);
        if ( v27 )
          j_j___libc_free_0_0(v27);
      }
    }
    while ( v24 != v25 );
    v25 = v44;
  }
  if ( v25 != &v46 )
    _libc_free(v25, v23);
  return v4;
}
