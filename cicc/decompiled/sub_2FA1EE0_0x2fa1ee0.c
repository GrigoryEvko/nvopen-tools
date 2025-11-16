// Function: sub_2FA1EE0
// Address: 0x2fa1ee0
//
__int64 __fastcall sub_2FA1EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // r12
  char v8; // r13
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // eax
  char v19; // si
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  __int64 v22; // r8
  unsigned __int8 *v23; // r13
  __int64 v24; // rax
  char v25; // si
  int v26; // edx
  unsigned __int8 **v27; // rax
  unsigned int v29; // edx
  __int64 v30; // r10
  __int64 v31; // r10
  unsigned __int8 **v32; // rax
  __int64 v33; // [rsp+8h] [rbp-158h]
  unsigned int v34; // [rsp+10h] [rbp-150h]
  __int64 v36; // [rsp+30h] [rbp-130h]
  char v38; // [rsp+46h] [rbp-11Ah]
  char v39; // [rsp+47h] [rbp-119h]
  __int64 v40; // [rsp+48h] [rbp-118h]
  __int64 v41[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v42; // [rsp+60h] [rbp-100h] BYREF
  __int64 v43; // [rsp+68h] [rbp-F8h]
  __int64 v44; // [rsp+70h] [rbp-F0h]
  unsigned int v45; // [rsp+78h] [rbp-E8h]
  __m128i v46; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+90h] [rbp-D0h]
  __int64 v48; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v49; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v50; // [rsp+B0h] [rbp-B0h]
  __int64 v51; // [rsp+B8h] [rbp-A8h] BYREF
  char v52; // [rsp+C0h] [rbp-A0h]
  int v53; // [rsp+C4h] [rbp-9Ch]
  __int64 v54; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v55; // [rsp+E8h] [rbp-78h]
  __int64 v56; // [rsp+F0h] [rbp-70h]
  __int64 v57; // [rsp+F8h] [rbp-68h]
  _BYTE *v58; // [rsp+100h] [rbp-60h]
  __int64 v59; // [rsp+108h] [rbp-58h]
  _BYTE v60[80]; // [rsp+110h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 56);
  v59 = 0x400000000LL;
  v41[1] = (__int64)&v54;
  v4 = a2 + 48;
  v5 = 0;
  v6 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = v60;
  v41[0] = (__int64)&v42;
  v40 = v4;
  if ( v3 == v4 )
    goto LABEL_31;
  v7 = 0;
  v8 = 0;
  v9 = v3;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v9 + 8);
      if ( !sub_B46AA0(v9 - 24) )
        break;
LABEL_36:
      v9 = v10;
      if ( v10 == v40 )
        goto LABEL_28;
    }
    if ( !v8 )
    {
      sub_2FA1890(&v46, v41, (unsigned __int8 *)(v9 - 24), v11, v12, v13);
      v7 = v47;
    }
    if ( v7 == v43 + 24LL * v45 )
      goto LABEL_35;
    v39 = *(_BYTE *)(v7 + 16);
    if ( v39 )
      goto LABEL_35;
    v8 = sub_DFAC70(*(_QWORD *)(a1 + 24), v9 - 24);
    if ( !v8 )
      goto LABEL_36;
    v36 = *(_QWORD *)(v7 + 8);
    v8 = sub_BCAC40(*(_QWORD *)(v36 + 8), 1);
    if ( !v8 )
      goto LABEL_36;
    v16 = (unsigned __int64)&v51;
    v49 = &v51;
    v48 = v36;
    v17 = 0x200000000LL;
    v50 = 0x200000000LL;
    v18 = *(_DWORD *)(v7 + 20);
    v19 = *(_BYTE *)(v7 + 17);
    v51 = v9 - 24;
    v53 = v18;
    v52 = v19;
    v20 = *(_QWORD *)(a1 + 16);
    LODWORD(v50) = 1;
    v21 = *(__int64 (**)())(*(_QWORD *)v20 + 104LL);
    if ( v21 != sub_2D56590
      && !((unsigned __int8 (__fastcall *)(__int64, bool))v21)(
            v20,
            (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 - 16) + 8LL) - 17 <= 1) )
    {
      if ( v49 != &v51 )
        _libc_free((unsigned __int64)v49);
LABEL_35:
      v8 = 0;
      goto LABEL_36;
    }
    if ( v10 == v40 )
    {
LABEL_24:
      v9 = v40;
      goto LABEL_25;
    }
    v38 = v8;
    while ( 1 )
    {
      v22 = v10 - 24;
      if ( !v10 )
        v22 = 0;
      v23 = (unsigned __int8 *)v22;
      if ( sub_B46AA0(v22) )
        goto LABEL_13;
      sub_2FA1890(&v46, v41, v23, v17, v14, v15);
      v7 = v47;
      v16 = 3LL * v45;
      if ( v47 == v43 + 24LL * v45 )
        break;
      v17 = v36;
      if ( v36 != *(_QWORD *)(v47 + 8) )
        break;
      if ( *(_BYTE *)(v47 + 16) )
      {
LABEL_13:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v10 == v40 )
          goto LABEL_24;
      }
      else
      {
        v24 = (unsigned int)v50;
        v25 = *(_BYTE *)(v47 + 17);
        v17 = *(unsigned int *)(v47 + 20);
        v26 = v50;
        if ( (unsigned int)v50 >= (unsigned __int64)HIDWORD(v50) )
        {
          v29 = v34;
          v17 <<= 32;
          LOBYTE(v29) = *(_BYTE *)(v47 + 17);
          v30 = v29;
          v16 = (unsigned int)v50 + 1LL;
          v31 = v17 | v30;
          v34 = v31;
          if ( HIDWORD(v50) < v16 )
          {
            v33 = v31;
            sub_C8D5F0((__int64)&v49, &v51, v16, 0x10u, v14, v15);
            v24 = (unsigned int)v50;
            v31 = v33;
          }
          v32 = (unsigned __int8 **)&v49[2 * v24];
          *v32 = v23;
          v32[1] = (unsigned __int8 *)v31;
          LODWORD(v50) = v50 + 1;
          goto LABEL_13;
        }
        v27 = (unsigned __int8 **)&v49[2 * (unsigned int)v50];
        if ( v27 )
        {
          *v27 = v23;
          *((_BYTE *)v27 + 8) = v25;
          *((_DWORD *)v27 + 3) = v17;
          v26 = v50;
        }
        v16 = (unsigned int)(v26 + 1);
        LODWORD(v50) = v16;
        v10 = *(_QWORD *)(v10 + 8);
        if ( v10 == v40 )
          goto LABEL_24;
      }
    }
    v9 = v10;
    v39 = v38;
LABEL_25:
    sub_2F9A860(a3, (unsigned __int64)&v48, v16, v17, v14, v15);
    if ( v49 != &v51 )
      _libc_free((unsigned __int64)v49);
    v8 = v39;
  }
  while ( v9 != v40 );
LABEL_28:
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  v6 = v55;
  v5 = 8LL * (unsigned int)v57;
LABEL_31:
  sub_C7D6A0(v6, v5, 8);
  return sub_C7D6A0(v43, 24LL * v45, 8);
}
