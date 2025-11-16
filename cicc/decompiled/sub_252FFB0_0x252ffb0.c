// Function: sub_252FFB0
// Address: 0x252ffb0
//
__int64 __fastcall sub_252FFB0(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, __int64, __int64 *),
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        char a8,
        unsigned __int8 (__fastcall *a9)(__int64, __int64, __int64),
        __int64 a10)
{
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 **v18; // rdi
  __int64 v19; // r14
  _BYTE *v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 v24; // rdx
  __int64 *v25; // r13
  __int64 *v26; // r14
  unsigned int v27; // eax
  __int64 *v28; // rbx
  unsigned int v29; // r15d
  __int64 *v30; // r12
  void (__fastcall *v31)(__int64 *, __int64 *, __int64); // rax
  __int64 *v33; // r13
  void (__fastcall *v34)(__int64 *, __int64 *, __int64); // rax
  unsigned __int64 v35; // rax
  _QWORD *v36; // r13
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  _QWORD *v43; // rax
  char v44; // dl
  __int64 v45; // r10
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 i; // r14
  __int64 v49; // rdx
  char v50; // dl
  char v51; // al
  __int64 v52; // r8
  __int64 v53; // r9
  _BYTE *v54; // rdi
  int v55; // esi
  unsigned __int64 v56; // r15
  __int64 v57; // r12
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // [rsp+0h] [rbp-230h]
  unsigned __int8 v61; // [rsp+10h] [rbp-220h]
  __int64 v62; // [rsp+18h] [rbp-218h]
  _BYTE *v63; // [rsp+18h] [rbp-218h]
  char v64; // [rsp+38h] [rbp-1F8h]
  __int64 v67; // [rsp+68h] [rbp-1C8h]
  char v68; // [rsp+7Fh] [rbp-1B1h] BYREF
  __int64 v69; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v70; // [rsp+88h] [rbp-1A8h]
  __int64 v71; // [rsp+90h] [rbp-1A0h]
  __int64 v72; // [rsp+98h] [rbp-198h]
  _BYTE *v73; // [rsp+A0h] [rbp-190h]
  __int64 v74; // [rsp+A8h] [rbp-188h]
  _BYTE v75[32]; // [rsp+B0h] [rbp-180h] BYREF
  _BYTE *v76; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-158h]
  _BYTE v78[128]; // [rsp+E0h] [rbp-150h] BYREF
  __int64 *v79; // [rsp+160h] [rbp-D0h] BYREF
  unsigned __int64 v80; // [rsp+168h] [rbp-C8h]
  __int64 v81; // [rsp+170h] [rbp-C0h] BYREF
  int v82; // [rsp+178h] [rbp-B8h]
  char v83; // [rsp+17Ch] [rbp-B4h]
  char v84; // [rsp+180h] [rbp-B0h] BYREF

  v12 = *(_QWORD *)(a1 + 104);
  v67 = a5;
  v64 = a6;
  v13 = *(_DWORD *)(a1 + 120);
  if ( !v13 )
    goto LABEL_4;
  v14 = (unsigned int)(v13 - 1);
  v15 = v14 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
  v16 = v12 + 56LL * v15;
  v17 = *(_QWORD *)v16;
  if ( a5 != *(_QWORD *)v16 )
  {
    v55 = 1;
    while ( v17 != -4096 )
    {
      a5 = (unsigned int)(v55 + 1);
      v15 = v14 & (v55 + v15);
      v16 = v12 + 56LL * v15;
      v17 = *(_QWORD *)v16;
      if ( v67 == *(_QWORD *)v16 )
        goto LABEL_3;
      v55 = a5;
    }
LABEL_4:
    v19 = *(_QWORD *)(v67 + 16);
    if ( !v19 )
      return 1;
    goto LABEL_5;
  }
LABEL_3:
  v18 = &v79;
  v79 = &v81;
  v80 = 0x100000000LL;
  if ( !*(_DWORD *)(v16 + 16) )
    goto LABEL_4;
  v23 = (__int64 *)(v16 + 8);
  sub_2512350((__int64)&v79, v23, v17, v14, a5);
  v25 = v79;
  a5 = 32LL * (unsigned int)v80;
  v26 = (__int64 *)((char *)v79 + a5);
  if ( v79 != (__int64 *)((char *)v79 + a5) )
  {
    do
    {
      v76 = (_BYTE *)a4;
      if ( !v25[2] )
        sub_4263D6(v18, v23, v24);
      v23 = (__int64 *)a1;
      v18 = (__int64 **)v25;
      v27 = ((__int64 (__fastcall *)(__int64 *, __int64, _BYTE **))v25[3])(v25, a1, &v76);
      if ( !(_BYTE)v27 )
      {
        v28 = v79;
        v29 = v27;
        v30 = &v79[4 * (unsigned int)v80];
        if ( v79 != v30 )
        {
          do
          {
            v31 = (void (__fastcall *)(__int64 *, __int64 *, __int64))*(v30 - 2);
            v30 -= 4;
            if ( v31 )
              v31(v30, v30, 3);
          }
          while ( v28 != v30 );
          v30 = v79;
        }
        if ( v30 != &v81 )
          _libc_free((unsigned __int64)v30);
        return v29;
      }
      v25 += 4;
    }
    while ( v25 != v26 );
    v33 = v79;
    a5 = 32LL * (unsigned int)v80;
    v26 = (__int64 *)((char *)v79 + a5);
    if ( v79 != (__int64 *)((char *)v79 + a5) )
    {
      do
      {
        v34 = (void (__fastcall *)(__int64 *, __int64 *, __int64))*(v26 - 2);
        v26 -= 4;
        if ( v34 )
          v34(v26, v26, 3);
      }
      while ( v33 != v26 );
      v26 = v79;
    }
  }
  if ( v26 == &v81 )
    goto LABEL_4;
  _libc_free((unsigned __int64)v26);
  v19 = *(_QWORD *)(v67 + 16);
  if ( !v19 )
    return 1;
LABEL_5:
  v79 = 0;
  v20 = v78;
  v77 = 0x1000000000LL;
  v80 = (unsigned __int64)&v84;
  v21 = v19;
  v22 = 0;
  v76 = v78;
  v81 = 16;
  v82 = 0;
  v83 = 1;
  while ( 1 )
  {
    *(_QWORD *)&v20[8 * v22] = v21;
    v22 = (unsigned int)(v77 + 1);
    LODWORD(v77) = v77 + 1;
    v21 = *(_QWORD *)(v21 + 8);
    if ( !v21 )
      break;
    if ( v22 + 1 > (unsigned __int64)HIDWORD(v77) )
    {
      sub_C8D5F0((__int64)&v76, v78, v22 + 1, 8u, a5, a6);
      v22 = (unsigned int)v77;
    }
    v20 = v76;
  }
  v35 = sub_25096F0((_QWORD *)(a4 + 72));
  v36 = (_QWORD *)v35;
  if ( v35 )
  {
    sub_250D230((unsigned __int64 *)&v69, v35, 4, 0);
    v36 = (_QWORD *)sub_251BBC0(a1, v69, v70, a4, 2, 0, 1);
  }
  LODWORD(v37) = v77;
  if ( !(_DWORD)v77 )
  {
LABEL_42:
    v29 = 1;
    goto LABEL_43;
  }
  while ( 1 )
  {
    v38 = *(_QWORD *)&v76[8 * (unsigned int)v37 - 8];
    LODWORD(v77) = v37 - 1;
    if ( (unsigned __int8)sub_250E2E0(*(_BYTE **)(v38 + 24)) )
    {
      if ( !v83 )
        goto LABEL_47;
      v43 = (_QWORD *)v80;
      v40 = HIDWORD(v81);
      v39 = (__int64 *)(v80 + 8LL * HIDWORD(v81));
      if ( (__int64 *)v80 != v39 )
      {
        while ( v38 != *v43 )
        {
          if ( v39 == ++v43 )
            goto LABEL_68;
        }
        goto LABEL_40;
      }
LABEL_68:
      if ( HIDWORD(v81) < (unsigned int)v81 )
      {
        ++HIDWORD(v81);
        *v39 = v38;
        v79 = (__int64 *)((char *)v79 + 1);
      }
      else
      {
LABEL_47:
        sub_C8CC70((__int64)&v79, v38, (__int64)v39, v40, v41, v42);
        if ( !v44 )
          goto LABEL_40;
      }
    }
    v68 = 0;
    v29 = sub_2522C50(a1, (unsigned __int64 *)v38, a4, v36, &v68, v64, a7);
    if ( (_BYTE)v29 )
      goto LABEL_40;
    if ( a8 )
    {
      if ( sub_BD2BE0(*(_QWORD *)(v38 + 24)) )
        goto LABEL_40;
      v45 = *(_QWORD *)(v38 + 24);
      if ( *(_BYTE *)v45 != 62 )
        goto LABEL_51;
    }
    else
    {
      v45 = *(_QWORD *)(v38 + 24);
      if ( *(_BYTE *)v45 != 62 )
        goto LABEL_51;
    }
    v49 = (*(_BYTE *)(v45 + 7) & 0x40) != 0 ? *(_QWORD *)(v45 - 8) : v45 - 32LL * (*(_DWORD *)(v45 + 4) & 0x7FFFFFF);
    if ( v38 == v49 )
      break;
LABEL_51:
    LOBYTE(v69) = 0;
    if ( !a2(a3, v38, &v69) )
      goto LABEL_43;
    if ( (_BYTE)v69 )
    {
      v37 = (unsigned int)v77;
      for ( i = *(_QWORD *)(*(_QWORD *)(v38 + 24) + 16LL); i; i = *(_QWORD *)(i + 8) )
      {
        if ( v37 + 1 > (unsigned __int64)HIDWORD(v77) )
        {
          sub_C8D5F0((__int64)&v76, v78, v37 + 1, 8u, v46, v47);
          v37 = (unsigned int)v77;
        }
        *(_QWORD *)&v76[8 * v37] = i;
        v37 = (unsigned int)(v77 + 1);
        LODWORD(v77) = v77 + 1;
      }
      goto LABEL_41;
    }
LABEL_40:
    LODWORD(v37) = v77;
LABEL_41:
    if ( !(_DWORD)v37 )
      goto LABEL_42;
  }
  v62 = v45;
  sub_AE6EC0((__int64)&v79, v38);
  if ( !v50 )
    goto LABEL_40;
  v69 = 0;
  v73 = v75;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0x400000000LL;
  v51 = sub_252FFA0(a1, v62, (__int64)&v69, a4, &v68, 1);
  v54 = v73;
  if ( !v51 )
  {
    if ( v73 != v75 )
      _libc_free((unsigned __int64)v73);
    sub_C7D6A0(v70, 8LL * (unsigned int)v72, 8);
    goto LABEL_51;
  }
  v63 = &v73[8 * (unsigned int)v74];
  if ( v63 == v73 )
  {
LABEL_87:
    if ( v54 != v75 )
      _libc_free((unsigned __int64)v54);
    sub_C7D6A0(v70, 8LL * (unsigned int)v72, 8);
    goto LABEL_40;
  }
  v60 = a4;
  v61 = v29;
  v56 = (unsigned __int64)v73;
  while ( 1 )
  {
    v57 = *(_QWORD *)(*(_QWORD *)v56 + 16LL);
    if ( v57 )
      break;
LABEL_85:
    v56 += 8LL;
    if ( v63 == (_BYTE *)v56 )
    {
      a4 = v60;
      v54 = v73;
      goto LABEL_87;
    }
  }
  while ( !a9 || a9(a10, v38, v57) )
  {
    v58 = (unsigned int)v77;
    v59 = (unsigned int)v77 + 1LL;
    if ( v59 > HIDWORD(v77) )
    {
      sub_C8D5F0((__int64)&v76, v78, v59, 8u, v52, v53);
      v58 = (unsigned int)v77;
    }
    *(_QWORD *)&v76[8 * v58] = v57;
    LODWORD(v77) = v77 + 1;
    v57 = *(_QWORD *)(v57 + 8);
    if ( !v57 )
      goto LABEL_85;
  }
  v29 = v61;
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  sub_C7D6A0(v70, 8LL * (unsigned int)v72, 8);
LABEL_43:
  if ( !v83 )
    _libc_free(v80);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  return v29;
}
