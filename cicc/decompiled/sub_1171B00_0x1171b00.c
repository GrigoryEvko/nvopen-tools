// Function: sub_1171B00
// Address: 0x1171b00
//
__int64 __fastcall sub_1171B00(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v6; // rcx
  unsigned int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rax
  char v11; // dl
  __int64 v12; // r13
  _BYTE *v13; // rax
  unsigned int v14; // r13d
  __int64 v16; // rdx
  __int64 **v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 *v22; // r14
  __int64 **v23; // r14
  __int64 **v24; // r13
  __int64 v25; // rdx
  __int64 **v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-168h]
  __int64 v30; // [rsp+8h] [rbp-168h]
  _BYTE *v31; // [rsp+8h] [rbp-168h]
  __int64 *v32; // [rsp+10h] [rbp-160h] BYREF
  __int64 v33; // [rsp+18h] [rbp-158h]
  _QWORD v34[16]; // [rsp+20h] [rbp-150h] BYREF
  __int64 v35; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v36; // [rsp+A8h] [rbp-C8h]
  __int64 v37; // [rsp+B0h] [rbp-C0h]
  int v38; // [rsp+B8h] [rbp-B8h]
  unsigned __int8 v39; // [rsp+BCh] [rbp-B4h]
  char v40; // [rsp+C0h] [rbp-B0h] BYREF

  v6 = 1;
  v32 = v34;
  v35 = 0;
  v37 = 16;
  v38 = 0;
  v39 = 1;
  v34[0] = a2;
  v36 = (__int64 *)&v40;
  v33 = 0x1000000001LL;
  v7 = 1;
  while ( v7 )
  {
    v8 = v32;
    v9 = v32[v7 - 1];
    LODWORD(v33) = v7 - 1;
    if ( !(_BYTE)v6 )
      goto LABEL_10;
    v10 = v36;
    a2 = HIDWORD(v37);
    v8 = &v36[HIDWORD(v37)];
    if ( v36 != v8 )
    {
      while ( v9 != *v10 )
      {
        if ( v8 == ++v10 )
          goto LABEL_9;
      }
      goto LABEL_8;
    }
LABEL_9:
    if ( HIDWORD(v37) < (unsigned int)v37 )
    {
      a2 = (unsigned int)++HIDWORD(v37);
      *v8 = v9;
      v6 = v39;
      ++v35;
    }
    else
    {
LABEL_10:
      a2 = v9;
      sub_C8CC70((__int64)&v35, v9, (__int64)v8, v6, a5, a6);
      v6 = v39;
      if ( !v11 )
        goto LABEL_8;
    }
    if ( HIDWORD(v37) - v38 == 16 )
    {
      v14 = 0;
      goto LABEL_15;
    }
    v12 = *(_QWORD *)(v9 + 16);
    if ( v12 )
    {
      do
      {
        v13 = *(_BYTE **)(v12 + 24);
        if ( *v13 != 84 )
        {
          LOBYTE(v6) = v39;
          v14 = 0;
          goto LABEL_15;
        }
        v16 = (unsigned int)v33;
        a5 = (unsigned int)v33 + 1LL;
        if ( a5 > HIDWORD(v33) )
        {
          a2 = (unsigned __int64)v34;
          v31 = *(_BYTE **)(v12 + 24);
          sub_C8D5F0((__int64)&v32, v34, (unsigned int)v33 + 1LL, 8u, a5, a6);
          v16 = (unsigned int)v33;
          v13 = v31;
        }
        v32[v16] = (__int64)v13;
        v7 = v33 + 1;
        LODWORD(v33) = v33 + 1;
        v12 = *(_QWORD *)(v12 + 8);
      }
      while ( v12 );
      v6 = v39;
    }
    else
    {
LABEL_8:
      v7 = v33;
    }
  }
  v17 = (__int64 **)v36;
  if ( (_BYTE)v6 )
  {
    v25 = HIDWORD(v37);
    v19 = &v36[HIDWORD(v37)];
    if ( v36 != v19 )
      goto LABEL_27;
    goto LABEL_38;
  }
  v18 = (unsigned int)v37;
  v19 = &v36[(unsigned int)v37];
  if ( v36 == v19 )
    goto LABEL_32;
LABEL_27:
  v20 = v36;
  while ( 1 )
  {
    v21 = *v20;
    v22 = v20;
    if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v19 == ++v20 )
      goto LABEL_30;
  }
  if ( v19 != v20 )
  {
    do
    {
      v29 = sub_ACADE0(*(__int64 ***)(v21 + 8));
      if ( *(_QWORD *)(v21 + 16) )
      {
        sub_10A5FE0(*(_QWORD *)(a1 + 40), v21);
        v27 = v29;
        if ( v29 == v21 )
          v27 = sub_ACADE0(*(__int64 ***)(v21 + 8));
        if ( !*(_QWORD *)(v27 + 16)
          && *(_BYTE *)v27 > 0x1Cu
          && (*(_BYTE *)(v27 + 7) & 0x10) == 0
          && (*(_BYTE *)(v21 + 7) & 0x10) != 0 )
        {
          v30 = v27;
          sub_BD6B90((unsigned __int8 *)v27, (unsigned __int8 *)v21);
          v27 = v30;
        }
        a2 = v27;
        sub_BD84D0(v21, v27);
      }
      v28 = v22 + 1;
      if ( v22 + 1 == v19 )
        break;
      while ( 1 )
      {
        v21 = *v28;
        v22 = v28;
        if ( (unsigned __int64)*v28 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v19 == ++v28 )
          goto LABEL_55;
      }
    }
    while ( v19 != v28 );
LABEL_55:
    v17 = (__int64 **)v36;
    LOBYTE(v6) = v39;
  }
LABEL_30:
  if ( (_BYTE)v6 )
  {
    v25 = HIDWORD(v37);
LABEL_38:
    v23 = &v17[v25];
    LOBYTE(v6) = 1;
    goto LABEL_33;
  }
  v18 = (unsigned int)v37;
LABEL_32:
  v23 = &v17[v18];
  LOBYTE(v6) = 0;
LABEL_33:
  if ( v17 == v23 )
    goto LABEL_36;
  while ( 1 )
  {
    a2 = (unsigned __int64)*v17;
    v24 = v17;
    if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v23 == ++v17 )
      goto LABEL_36;
  }
  if ( v23 == v17 )
  {
LABEL_36:
    v14 = 1;
  }
  else
  {
    do
    {
      sub_F207A0(a1, (__int64 *)a2);
      v26 = v24 + 1;
      if ( v24 + 1 == v23 )
        break;
      while ( 1 )
      {
        a2 = (unsigned __int64)*v26;
        v24 = v26;
        if ( (unsigned __int64)*v26 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v23 == ++v26 )
          goto LABEL_43;
      }
    }
    while ( v23 != v26 );
LABEL_43:
    LOBYTE(v6) = v39;
    v14 = 1;
  }
LABEL_15:
  if ( !(_BYTE)v6 )
    _libc_free(v36, a2);
  if ( v32 != v34 )
    _libc_free(v32, a2);
  return v14;
}
