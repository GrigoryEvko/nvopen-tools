// Function: sub_25230B0
// Address: 0x25230b0
//
__int64 __fastcall sub_25230B0(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64 *),
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        char a8)
{
  int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rdx
  _BYTE **v15; // rdi
  signed __int64 v16; // r15
  __int64 v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // rax
  _BYTE *v20; // rdi
  int v21; // r15d
  __int64 v22; // rdx
  unsigned __int8 *v23; // r13
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rsi
  bool v28; // al
  __int64 *v29; // rsi
  __int64 v30; // rdx
  _BYTE *v31; // r12
  _BYTE *v32; // r14
  unsigned int v33; // eax
  _BYTE *v34; // rbx
  unsigned int v35; // r15d
  _BYTE *v36; // r12
  void (__fastcall *v37)(_BYTE *, _BYTE *, __int64); // rax
  unsigned __int8 *v39; // rdi
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 j; // rax
  _BYTE *v43; // r13
  void (__fastcall *v44)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v45; // r13
  int i; // esi
  unsigned __int64 v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int8 *v52; // rdi
  unsigned __int64 v53; // rax
  __int64 v54; // r13
  unsigned int v55; // r12d
  unsigned int v56; // ebx
  __int64 v57; // rax
  __int64 v58; // r15
  bool v59; // al
  char v60; // r13
  __int64 v61; // r8
  __int64 v62; // r9
  int v63; // [rsp+10h] [rbp-130h]
  __int64 v64; // [rsp+28h] [rbp-118h]
  char v67; // [rsp+46h] [rbp-FAh]
  unsigned __int8 *v70; // [rsp+60h] [rbp-E0h] BYREF
  char *v71; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int v72; // [rsp+70h] [rbp-D0h]
  char v73; // [rsp+78h] [rbp-C8h] BYREF
  unsigned __int8 *v74; // [rsp+80h] [rbp-C0h]
  char *v75; // [rsp+88h] [rbp-B8h] BYREF
  __int64 v76; // [rsp+90h] [rbp-B0h]
  char v77; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v78; // [rsp+A0h] [rbp-A0h] BYREF
  _BYTE *v79; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v80; // [rsp+B0h] [rbp-90h]
  _BYTE v81[8]; // [rsp+B8h] [rbp-88h] BYREF
  _BYTE *v82; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v83; // [rsp+C8h] [rbp-78h]
  _BYTE v84[112]; // [rsp+D0h] [rbp-70h] BYREF

  v67 = a5;
  if ( (_BYTE)a5 && (*(_BYTE *)(a4 + 32) & 0xFu) - 7 > 1 )
    return 0;
  v9 = *(_DWORD *)(a1 + 120);
  v10 = *(_QWORD *)(a1 + 104);
  if ( !v9 )
    goto LABEL_6;
  v11 = (unsigned int)(v9 - 1);
  v12 = v11 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v13 = v10 + 56LL * v12;
  v14 = *(_QWORD *)v13;
  if ( a4 != *(_QWORD *)v13 )
  {
    for ( i = 1; ; i = a5 )
    {
      if ( v14 == -4096 )
        goto LABEL_6;
      a5 = (unsigned int)(i + 1);
      v12 = v11 & (i + v12);
      v13 = v10 + 56LL * v12;
      v14 = *(_QWORD *)v13;
      if ( a4 == *(_QWORD *)v13 )
        break;
    }
  }
  v15 = &v82;
  v82 = v84;
  v83 = 0x100000000LL;
  if ( !*(_DWORD *)(v13 + 16) )
    goto LABEL_6;
  v29 = (__int64 *)(v13 + 8);
  sub_2512350((__int64)&v82, v29, v14, v11, a5);
  v31 = v82;
  a5 = 32LL * (unsigned int)v83;
  v32 = &v82[a5];
  if ( v82 == &v82[a5] )
    goto LABEL_59;
  while ( 1 )
  {
    v78 = a6;
    if ( !*((_QWORD *)v31 + 2) )
      sub_4263D6(v15, v29, v30);
    v29 = (__int64 *)a1;
    v15 = (_BYTE **)v31;
    v33 = (*((__int64 (__fastcall **)(_BYTE *, __int64, __int64 *))v31 + 3))(v31, a1, &v78);
    if ( !(_BYTE)v33 )
      break;
    v31 += 32;
    if ( v31 == v32 )
    {
      v43 = v82;
      a5 = 32LL * (unsigned int)v83;
      v32 = &v82[a5];
      if ( v82 != &v82[a5] )
      {
        do
        {
          v44 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v32 - 2);
          v32 -= 32;
          if ( v44 )
            v44(v32, v32, 3);
        }
        while ( v43 != v32 );
        v32 = v82;
      }
LABEL_59:
      if ( v32 != v84 )
        _libc_free((unsigned __int64)v32);
LABEL_6:
      v16 = 0;
      v17 = *(_QWORD *)(a4 + 16);
      v82 = v84;
      v83 = 0x800000000LL;
      v18 = v17;
      if ( !v17 )
        return 1;
      do
      {
        v18 = *(_QWORD *)(v18 + 8);
        ++v16;
      }
      while ( v18 );
      v19 = v84;
      if ( v16 > 8 )
      {
        sub_C8D5F0((__int64)&v82, v84, v16, 8u, a5, a6);
        v19 = &v82[8 * (unsigned int)v83];
      }
      do
      {
        *v19 = v17;
        v17 = *(_QWORD *)(v17 + 8);
        ++v19;
      }
      while ( v17 );
      v20 = v82;
      LODWORD(v83) = v83 + v16;
      if ( (_DWORD)v83 )
      {
        v64 = a1;
        v21 = 0;
        v22 = 0;
        while ( 1 )
        {
          v23 = *(unsigned __int8 **)&v20[8 * v22];
          if ( a8 || !(unsigned __int8)sub_2522C50(v64, (unsigned __int64 *)v23, a6, 0, a7, 1, 1) )
            break;
LABEL_51:
          LODWORD(j) = v83;
LABEL_52:
          v22 = (unsigned int)(v21 + 1);
          v20 = v82;
          v21 = v22;
          if ( (unsigned int)v22 >= (unsigned int)j )
            goto LABEL_53;
        }
        v24 = *((_QWORD *)v23 + 3);
        if ( *(_BYTE *)v24 == 5 && sub_AC35E0(*((_QWORD *)v23 + 3)) && *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 14 )
        {
          v45 = *(_QWORD *)(v24 + 16);
          for ( j = (unsigned int)v83; v45; v45 = *(_QWORD *)(v45 + 8) )
          {
            if ( j + 1 > (unsigned __int64)HIDWORD(v83) )
            {
              sub_C8D5F0((__int64)&v82, v84, j + 1, 8u, v25, v26);
              j = (unsigned int)v83;
            }
            *(_QWORD *)&v82[8 * j] = v45;
            j = (unsigned int)(v83 + 1);
            LODWORD(v83) = v83 + 1;
          }
          goto LABEL_52;
        }
        v27 = (__int64)v23;
        sub_E33C60((__int64 *)&v70, (__int64)v23);
        if ( !v70 )
        {
          if ( **((_BYTE **)v23 + 3) != 4 )
            goto LABEL_43;
          goto LABEL_49;
        }
        if ( v72 )
        {
          if ( (v70[7] & 0x40) != 0 )
            v39 = (unsigned __int8 *)*((_QWORD *)v70 - 1);
          else
            v39 = &v70[-32 * (*((_DWORD *)v70 + 1) & 0x7FFFFFF)];
          v23 = &v39[32 * *(unsigned int *)v71];
        }
        else if ( !sub_B491E0((__int64)v70) )
        {
          v28 = v23 == v70 - 32;
          goto LABEL_41;
        }
        v40 = *((_QWORD *)v23 + 3);
        if ( *(_BYTE *)v40 == 5 )
        {
          v41 = *(_QWORD *)(v40 + 16);
          if ( v41 )
          {
            if ( !*(_QWORD *)(v41 + 8) && sub_AC35E0(*((_QWORD *)v23 + 3)) )
              v23 = *(unsigned __int8 **)(v40 + 16);
          }
        }
        v28 = *(_DWORD *)v71 == (unsigned int)((v23 - &v70[-32 * (*((_DWORD *)v70 + 1) & 0x7FFFFFF)]) >> 5);
LABEL_41:
        if ( !v28 )
        {
          if ( v67 )
            goto LABEL_43;
          goto LABEL_49;
        }
        v47 = *(_QWORD *)(a4 + 104);
        LODWORD(v53) = sub_25093E0(&v70);
        v52 = v70;
        v53 = (unsigned int)v53;
        if ( (unsigned int)v53 > v47 )
          v53 = v47;
        if ( v53 )
        {
          v54 = 0;
          v55 = 1;
          v63 = v21;
          v56 = v53;
          while ( 1 )
          {
            v50 = v72;
            if ( !v72 )
            {
              v59 = sub_B491E0((__int64)v52);
              v52 = v70;
              if ( !v59 )
                break;
            }
            v49 = v54 + 1;
            v57 = *(int *)&v71[4 * v54 + 4];
            if ( (int)v57 >= 0 )
            {
              v27 = *((_DWORD *)v52 + 1) & 0x7FFFFFF;
              v58 = *(_QWORD *)&v52[32 * (v57 - v27)];
              if ( v58 )
                goto LABEL_79;
            }
LABEL_84:
            v54 = v49;
            if ( v55 >= v56 )
            {
              v21 = v63;
              goto LABEL_96;
            }
            ++v55;
          }
          v49 = *((_DWORD *)v70 + 1) & 0x7FFFFFF;
          v58 = *(_QWORD *)&v70[32 * (v54 - v49)];
          if ( v58 )
          {
LABEL_79:
            if ( (*(_BYTE *)(a4 + 2) & 1) != 0 )
              sub_B2C6D0(a4, v27, v48, v49);
            if ( *(_QWORD *)(v58 + 8) != *(_QWORD *)(*(_QWORD *)(a4 + 96) + 40 * v54 + 8) )
              goto LABEL_43;
            v52 = v70;
          }
          v49 = v54 + 1;
          goto LABEL_84;
        }
LABEL_96:
        v74 = v52;
        v76 = 0;
        v75 = &v77;
        if ( v72 )
        {
          sub_2506A60((__int64)&v75, (__int64)&v71, v48, v49, v50, v51);
          v79 = v81;
          v78 = (__int64)v74;
          v80 = 0;
          if ( (_DWORD)v76 )
            sub_2506900((__int64)&v79, &v75, (__int64)v81, (unsigned int)v76, v61, v62);
        }
        else
        {
          v78 = (__int64)v52;
          v79 = v81;
          v80 = 0;
        }
        v60 = a2(a3, &v78);
        if ( v79 != v81 )
          _libc_free((unsigned __int64)v79);
        if ( v75 != &v77 )
          _libc_free((unsigned __int64)v75);
        if ( v60 )
        {
LABEL_49:
          if ( v71 != &v73 )
            _libc_free((unsigned __int64)v71);
          goto LABEL_51;
        }
LABEL_43:
        if ( v71 != &v73 )
          _libc_free((unsigned __int64)v71);
        v20 = v82;
        v35 = 0;
      }
      else
      {
LABEL_53:
        v35 = 1;
      }
      if ( v20 != v84 )
        _libc_free((unsigned __int64)v20);
      return v35;
    }
  }
  v34 = v82;
  v35 = v33;
  v36 = &v82[32 * (unsigned int)v83];
  if ( v82 != v36 )
  {
    do
    {
      v37 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)v36 - 2);
      v36 -= 32;
      if ( v37 )
        v37(v36, v36, 3);
    }
    while ( v34 != v36 );
    v36 = v82;
  }
  if ( v36 != v84 )
    _libc_free((unsigned __int64)v36);
  return v35;
}
