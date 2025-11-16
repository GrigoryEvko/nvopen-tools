// Function: sub_D75690
// Address: 0xd75690
//
unsigned __int64 *__fastcall sub_D75690(__int64 *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  unsigned __int64 *v5; // r10
  __int64 v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // r13
  __int64 *v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 *v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int64 v18; // rdx
  unsigned __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r15
  unsigned __int64 v22; // rdx
  __int64 v23; // r8
  unsigned __int64 v24; // r15
  unsigned __int64 *v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  char *v28; // rax
  unsigned __int64 *result; // rax
  __int64 v30; // rdx
  unsigned __int64 *v31; // r12
  unsigned __int64 *i; // r13
  unsigned __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // [rsp+0h] [rbp-6E0h]
  __int64 v37; // [rsp+0h] [rbp-6E0h]
  __int64 v38; // [rsp+0h] [rbp-6E0h]
  unsigned __int64 *v41; // [rsp+20h] [rbp-6C0h]
  unsigned __int64 *v43; // [rsp+40h] [rbp-6A0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-698h]
  _BYTE v45[64]; // [rsp+50h] [rbp-690h] BYREF
  unsigned __int64 *v46; // [rsp+90h] [rbp-650h] BYREF
  __int64 v47; // [rsp+98h] [rbp-648h]
  _BYTE v48[64]; // [rsp+A0h] [rbp-640h] BYREF
  unsigned __int64 *v49; // [rsp+E0h] [rbp-600h] BYREF
  __int64 v50; // [rsp+E8h] [rbp-5F8h]
  _BYTE v51[64]; // [rsp+F0h] [rbp-5F0h] BYREF
  _BYTE v52[704]; // [rsp+130h] [rbp-5B0h] BYREF
  _BYTE *v53; // [rsp+3F0h] [rbp-2F0h] BYREF
  __int64 v54; // [rsp+3F8h] [rbp-2E8h]
  _BYTE v55[288]; // [rsp+400h] [rbp-2E0h] BYREF
  _QWORD v56[2]; // [rsp+520h] [rbp-1C0h] BYREF
  char v57; // [rsp+530h] [rbp-1B0h] BYREF
  char v58; // [rsp+650h] [rbp-90h] BYREF
  char *v59; // [rsp+658h] [rbp-88h]
  __int64 v60; // [rsp+660h] [rbp-80h]
  char v61; // [rsp+668h] [rbp-78h] BYREF

  v5 = a2;
  v6 = a4;
  v7 = a1;
  v8 = 2 * a3;
  v43 = (unsigned __int64 *)v45;
  v9 = (__int64)&a2[v8];
  v44 = 0x400000000LL;
  v46 = (unsigned __int64 *)v48;
  v47 = 0x400000000LL;
  v49 = (unsigned __int64 *)v51;
  v50 = 0x400000000LL;
  if ( a2 == &a2[v8] )
    goto LABEL_17;
  v10 = (__int64 *)a2;
  v11 = (__int64 *)&a2[v8];
  v41 = a2;
  do
  {
    while ( 1 )
    {
      v34 = *v10;
      v15 = v10[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v10[1] & 4) == 0 )
        break;
      v16 = (unsigned int)v44;
      v17 = v15 | 4;
      v18 = (unsigned int)v44 + 1LL;
      if ( v18 > HIDWORD(v44) )
      {
        a2 = (unsigned __int64 *)v45;
        v38 = *v10;
        sub_C8D5F0((__int64)&v43, v45, v18, 0x10u, v34, v9);
        v16 = (unsigned int)v44;
        v34 = v38;
      }
      v19 = &v43[2 * v16];
      *v19 = v34;
      v19[1] = v17;
      v20 = (unsigned int)v47;
      v21 = v10[1];
      v22 = (unsigned int)v47 + 1LL;
      LODWORD(v44) = v44 + 1;
      v23 = *v10;
      v24 = v21 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v22 > HIDWORD(v47) )
      {
        a2 = (unsigned __int64 *)v48;
        v37 = *v10;
        sub_C8D5F0((__int64)&v46, v48, v22, 0x10u, v23, v9);
        v20 = (unsigned int)v47;
        v23 = v37;
      }
      v10 += 2;
      v25 = &v46[2 * v20];
      *v25 = v23;
      v25[1] = v24;
      LODWORD(v47) = v47 + 1;
      if ( v11 == v10 )
        goto LABEL_12;
    }
    v12 = (unsigned int)v50;
    v13 = (unsigned int)v50 + 1LL;
    if ( v13 > HIDWORD(v50) )
    {
      a2 = (unsigned __int64 *)v51;
      v36 = *v10;
      sub_C8D5F0((__int64)&v49, v51, v13, 0x10u, v34, v9);
      v12 = (unsigned int)v50;
      v34 = v36;
    }
    v10 += 2;
    v14 = (__int64 *)&v49[2 * v12];
    *v14 = v34;
    v14[1] = v15;
    LODWORD(v50) = v50 + 1;
  }
  while ( v11 != v10 );
LABEL_12:
  v26 = (unsigned int)v44;
  v7 = a1;
  v5 = v41;
  v6 = a4;
  if ( !(_DWORD)v44 )
  {
LABEL_17:
    if ( a5 )
    {
      sub_B26290((__int64)&v53, v5, a3, 1u);
      sub_B24D40(v6, (__int64)&v53, 0);
      sub_B1A8B0((__int64)&v53, (__int64)&v53);
    }
    v27 = v55;
    v53 = 0;
    v54 = 1;
    do
    {
      *v27 = -4096;
      v27 += 9;
    }
    while ( v27 != v56 );
    v28 = &v57;
    v56[0] = 0;
    v56[1] = 1;
    do
    {
      *(_QWORD *)v28 = -4096;
      v28 += 72;
    }
    while ( v28 != &v58 );
    a2 = v49;
    v59 = &v61;
    v60 = 0x400000000LL;
    v58 = 0;
    sub_D6FF50(v7, (__int64)v49, (unsigned int)v50, v6, (__int64)&v53, v9);
    sub_B1A8B0((__int64)&v53, (__int64)a2);
    v26 = (unsigned int)v44;
  }
  else if ( (_DWORD)v50 )
  {
    if ( a5 )
    {
      sub_B26980(a4, v41, a3, v46, (unsigned int)v47);
    }
    else
    {
      v54 = 0;
      v53 = v55;
      sub_B26980(a4, v55, 0, v46, (unsigned int)v47);
      if ( v53 != v55 )
        _libc_free(v53, v55);
    }
    sub_B26290((__int64)v52, v46, (unsigned int)v47, 0);
    sub_D6FF50(a1, (__int64)v49, (unsigned int)v50, a4, (__int64)v52, v35);
    sub_B26290((__int64)&v53, v43, (unsigned int)v44, 1u);
    a2 = (unsigned __int64 *)&v53;
    sub_B24D40(a4, (__int64)&v53, 0);
    sub_B1A8B0((__int64)a2, (__int64)a2);
    sub_B1A8B0((__int64)v52, (__int64)&v53);
    v26 = (unsigned int)v44;
  }
  else if ( a5 )
  {
    a2 = v43;
    sub_B26780(a4, v43, (unsigned int)v44);
    v26 = (unsigned int)v44;
  }
  result = v43;
  v30 = 2 * v26;
  v31 = &v43[v30];
  for ( i = v43; v31 != i; result = (unsigned __int64 *)sub_D6D7F0(v7, (__int64)a2, v33 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v33 = i[1];
    a2 = (unsigned __int64 *)*i;
    i += 2;
  }
  if ( v49 != (unsigned __int64 *)v51 )
    result = (unsigned __int64 *)_libc_free(v49, a2);
  if ( v46 != (unsigned __int64 *)v48 )
    result = (unsigned __int64 *)_libc_free(v46, a2);
  if ( v43 != (unsigned __int64 *)v45 )
    return (unsigned __int64 *)_libc_free(v43, a2);
  return result;
}
