// Function: sub_27B06D0
// Address: 0x27b06d0
//
void __fastcall sub_27B06D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // rcx
  char *v25; // r13
  __int64 v26; // r14
  __int64 v27; // rbx
  unsigned __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rbx
  char *v33; // rax
  unsigned __int64 v34; // rdi
  int v35; // r12d
  char *v36; // rsi
  unsigned __int64 v37; // r12
  size_t v38; // r12
  const void *v39; // rdi
  __int64 v40; // rdx
  size_t v41; // rdx
  void *v42; // rsi
  __int64 v43; // rdx
  size_t v44; // rdx
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // [rsp+10h] [rbp-130h]
  __int64 v52; // [rsp+28h] [rbp-118h]
  __int64 v53; // [rsp+38h] [rbp-108h]
  __int64 v54; // [rsp+48h] [rbp-F8h] BYREF
  void *s2; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v56; // [rsp+58h] [rbp-E8h]
  char v57; // [rsp+60h] [rbp-E0h] BYREF
  void *v58; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v59; // [rsp+88h] [rbp-B8h]
  char v60; // [rsp+90h] [rbp-B0h] BYREF
  void *v61; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-88h]
  _BYTE v63[32]; // [rsp+C0h] [rbp-80h] BYREF
  void *v64; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v65; // [rsp+E8h] [rbp-58h]
  _BYTE v66[80]; // [rsp+F0h] [rbp-50h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = a1;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v53 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 8) = sub_C7D670(96LL * v6, 8);
  if ( !v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
    {
      qword_4FFC5B0 = 0;
      qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
      qword_4FFC5D0 = (__int64)algn_4FFC5E0;
      qword_4FFC5D8 = 0x400000000LL;
      qword_4FFC5A8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC580);
    }
    v61 = v63;
    v62 = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5A8 )
      sub_27ABF90((__int64)&v61, (__int64)&qword_4FFC5A0, v7, v8, v9, v10);
    v64 = v66;
    v65 = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5D8 )
    {
      sub_27AC1D0((__int64)&v64, (__int64)&qword_4FFC5D0, v7, v8, v9, v10);
      v45 = *(_QWORD *)(a1 + 8);
      v49 = 96LL * *(unsigned int *)(a1 + 24);
      v47 = v45 + v49;
      if ( v45 != v45 + v49 )
        goto LABEL_82;
    }
    else
    {
      v45 = *(_QWORD *)(a1 + 8);
      v46 = 96LL * *(unsigned int *)(a1 + 24);
      v47 = v45 + v46;
      if ( v45 == v45 + v46 )
      {
LABEL_88:
        if ( v61 != v63 )
          _libc_free((unsigned __int64)v61);
        return;
      }
      do
      {
LABEL_82:
        while ( 1 )
        {
          if ( v45 )
          {
            *(_DWORD *)(v45 + 8) = 0;
            *(_QWORD *)v45 = v45 + 16;
            *(_DWORD *)(v45 + 12) = 4;
            if ( (_DWORD)v62 )
              sub_27ABF90(v45, (__int64)&v61, v7, v8, v9, v10);
            *(_DWORD *)(v45 + 56) = 0;
            *(_QWORD *)(v45 + 48) = v45 + 64;
            *(_DWORD *)(v45 + 60) = 4;
            if ( (_DWORD)v65 )
              break;
          }
          v45 += 96;
          if ( v45 == v47 )
            goto LABEL_86;
        }
        v48 = v45 + 48;
        v45 += 96;
        sub_27AC1D0(v48, (__int64)&v64, v7, v8, v9, v10);
      }
      while ( v45 != v47 );
    }
LABEL_86:
    if ( v64 != v66 )
      _libc_free((unsigned __int64)v64);
    goto LABEL_88;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v50 = 96 * v4;
  v11 = 96 * v4 + v5;
  if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
  {
    qword_4FFC5B0 = 0;
    qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
    qword_4FFC5D0 = (__int64)algn_4FFC5E0;
    qword_4FFC5D8 = 0x400000000LL;
    qword_4FFC5A8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC580);
  }
  v61 = v63;
  v62 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5A8 )
    sub_27ABF90((__int64)&v61, (__int64)&qword_4FFC5A0, v7, v8, v9, v10);
  v12 = (unsigned int)qword_4FFC5D8;
  v64 = v66;
  v65 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5D8 )
    sub_27AC1D0((__int64)&v64, (__int64)&qword_4FFC5D0, v7, v8, v9, (unsigned int)qword_4FFC5D8);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = 96LL * *(unsigned int *)(a1 + 24);
  if ( v13 != v13 + v14 )
  {
    v15 = v13 + v14;
    do
    {
      while ( 1 )
      {
        if ( v13 )
        {
          *(_DWORD *)(v13 + 8) = 0;
          *(_QWORD *)v13 = v13 + 16;
          *(_DWORD *)(v13 + 12) = 4;
          v16 = (unsigned int)v62;
          if ( (_DWORD)v62 )
            sub_27ABF90(v13, (__int64)&v61, v14, v8, (unsigned int)v62, v12);
          *(_DWORD *)(v13 + 56) = 0;
          *(_QWORD *)(v13 + 48) = v13 + 64;
          *(_DWORD *)(v13 + 60) = 4;
          if ( (_DWORD)v65 )
            break;
        }
        v13 += 96;
        if ( v15 == v13 )
          goto LABEL_17;
      }
      v17 = v13 + 48;
      v13 += 96;
      sub_27AC1D0(v17, (__int64)&v64, v14, v8, v16, v12);
    }
    while ( v15 != v13 );
LABEL_17:
    v3 = a1;
  }
  sub_27ABC80(&v61);
  if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
  {
    qword_4FFC5B0 = 0;
    qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
    qword_4FFC5D0 = (__int64)algn_4FFC5E0;
    qword_4FFC5D8 = 0x400000000LL;
    qword_4FFC5A8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC580);
  }
  s2 = &v57;
  v56 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5A8 )
    sub_27ABF90((__int64)&s2, (__int64)&qword_4FFC5A0, v18, v19, v20, v21);
  v22 = (unsigned int)qword_4FFC5D8;
  v58 = &v60;
  v59 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5D8 )
    sub_27AC1D0((__int64)&v58, (__int64)&qword_4FFC5D0, v18, (unsigned int)qword_4FFC5D8, v20, v21);
  if ( !byte_4FFC508 && (unsigned int)sub_2207590((__int64)&byte_4FFC508) )
  {
    qword_4FFC530 = 1;
    qword_4FFC520 = (__int64)&qword_4FFC530;
    qword_4FFC550 = (__int64)algn_4FFC560;
    qword_4FFC558 = 0x400000000LL;
    qword_4FFC528 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC520, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC508);
  }
  v23 = (unsigned int)qword_4FFC528;
  v61 = v63;
  v62 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC528 )
    sub_27ABF90((__int64)&v61, (__int64)&qword_4FFC520, (unsigned int)qword_4FFC528, v22, v20, v21);
  v64 = v66;
  v65 = 0x400000000LL;
  if ( (_DWORD)qword_4FFC558 )
    sub_27AC1D0((__int64)&v64, (__int64)&qword_4FFC550, v23, v22, v20, v21);
  v24 = &v54;
  v25 = (char *)(v53 + 16);
  v26 = v53;
  if ( v11 != v53 )
  {
    v52 = v3;
    do
    {
      v27 = *(unsigned int *)(v26 + 8);
      if ( v27 != (unsigned int)v56 )
        goto LABEL_31;
      v38 = 8 * v27;
      if ( 8 * v27 )
      {
        v39 = *(const void **)v26;
        if ( memcmp(*(const void **)v26, s2, 8 * v27) )
        {
          if ( v27 != (unsigned int)v62 )
            goto LABEL_32;
          v42 = v61;
LABEL_61:
          if ( memcmp(v39, v42, v38) )
            goto LABEL_32;
          goto LABEL_55;
        }
      }
      v40 = *(unsigned int *)(v26 + 56);
      if ( v40 == (unsigned int)v59 )
      {
        v41 = 8 * v40;
        v37 = *(_QWORD *)(v26 + 48);
        if ( !v41 || !memcmp(*(const void **)(v26 + 48), v58, v41) )
          goto LABEL_42;
        if ( v27 != (unsigned int)v62 )
          goto LABEL_32;
      }
      else
      {
LABEL_31:
        if ( v27 != (unsigned int)v62 )
          goto LABEL_32;
      }
      v42 = v61;
      v39 = *(const void **)v26;
      v38 = 8 * v27;
      if ( 8 * v27 )
        goto LABEL_61;
LABEL_55:
      v43 = *(unsigned int *)(v26 + 56);
      if ( v43 == (unsigned int)v65 )
      {
        v44 = 8 * v43;
        v37 = *(_QWORD *)(v26 + 48);
        if ( !v44 || !memcmp(*(const void **)(v26 + 48), v64, v44) )
          goto LABEL_42;
      }
LABEL_32:
      sub_27B0210(v52, (const void **)v26, &v54, (__int64)v24, v20, v21);
      v32 = v54;
      if ( v54 != v26 )
      {
        v33 = *(char **)v26;
        if ( v25 != *(char **)v26 )
        {
          v28 = v54 + 16;
          if ( *(_QWORD *)v54 != v54 + 16 )
          {
            _libc_free(*(_QWORD *)v54);
            v33 = *(char **)v26;
          }
          *(_QWORD *)v32 = v33;
          *(_DWORD *)(v32 + 8) = *(_DWORD *)(v26 + 8);
          *(_DWORD *)(v32 + 12) = *(_DWORD *)(v26 + 12);
          *(_QWORD *)v26 = v25;
          *(_DWORD *)(v26 + 12) = 0;
          *(_DWORD *)(v26 + 8) = 0;
          goto LABEL_41;
        }
        v28 = *(unsigned int *)(v26 + 8);
        v34 = *(unsigned int *)(v54 + 8);
        v35 = *(_DWORD *)(v26 + 8);
        if ( v28 <= v34 )
        {
          if ( *(_DWORD *)(v26 + 8) )
            memmove(*(void **)v54, v25, 8 * v28);
        }
        else if ( v28 > *(unsigned int *)(v54 + 12) )
        {
          *(_DWORD *)(v54 + 8) = 0;
          sub_C8D5F0(v32, (const void *)(v32 + 16), v28, 8u, v30, v31);
          v34 = 0;
          v28 = 8LL * *(unsigned int *)(v26 + 8);
          v36 = *(char **)v26;
          if ( *(_QWORD *)v26 != v28 + *(_QWORD *)v26 )
            goto LABEL_39;
        }
        else
        {
          v29 = 8 * v34;
          v36 = v25;
          if ( *(_DWORD *)(v54 + 8) )
          {
            memmove(*(void **)v54, v25, v29);
            v29 = 8 * v34;
            v33 = *(char **)v26;
            v28 = *(unsigned int *)(v26 + 8);
            v36 = (char *)(*(_QWORD *)v26 + 8 * v34);
            v34 *= 8LL;
          }
          v28 *= 8LL;
          if ( v36 != &v33[v28] )
LABEL_39:
            memcpy((void *)(v34 + *(_QWORD *)v32), v36, v28 - v34);
        }
        *(_DWORD *)(v32 + 8) = v35;
        *(_DWORD *)(v26 + 8) = 0;
      }
LABEL_41:
      sub_27AC070(v32 + 48, (char **)(v26 + 48), v28, v29, v30, v31);
      ++*(_DWORD *)(v52 + 16);
      v37 = *(_QWORD *)(v26 + 48);
LABEL_42:
      if ( v37 != v26 + 64 )
        _libc_free(v37);
      if ( v25 != *(char **)v26 )
        _libc_free(*(_QWORD *)v26);
      v26 += 96;
      v25 += 96;
    }
    while ( v11 != v26 );
  }
  sub_27ABC80(&v61);
  sub_27ABC80(&s2);
  sub_C7D6A0(v53, v50, 8);
}
