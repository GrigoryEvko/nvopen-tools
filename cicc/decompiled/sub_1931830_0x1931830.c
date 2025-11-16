// Function: sub_1931830
// Address: 0x1931830
//
void __fastcall sub_1931830(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  char **v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  char **v10; // r12
  int v11; // r9d
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rcx
  __int64 v16; // rdx
  char *v17; // r14
  char **v18; // r15
  __int64 v19; // rbx
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rbx
  char *v25; // rax
  unsigned __int64 v26; // r13
  size_t v27; // r13
  char *v28; // rdi
  __int64 v29; // rdx
  size_t v30; // rdx
  void *v31; // rsi
  __int64 v32; // rdx
  size_t v33; // rdx
  unsigned __int64 v34; // rdi
  int v35; // r13d
  char *v36; // rsi
  __int64 v37; // r12
  __int64 v38; // r13
  __int64 v39; // rdi
  char **v40; // [rsp+0h] [rbp-140h]
  __int64 v41; // [rsp+20h] [rbp-120h]
  __int64 v42; // [rsp+48h] [rbp-F8h] BYREF
  void *s2; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+58h] [rbp-E8h]
  _BYTE v45[32]; // [rsp+60h] [rbp-E0h] BYREF
  void *v46; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+88h] [rbp-B8h]
  _BYTE v48[32]; // [rsp+90h] [rbp-B0h] BYREF
  void *v49; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-88h]
  _BYTE v51[32]; // [rsp+C0h] [rbp-80h] BYREF
  void *v52; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v53; // [rsp+E8h] [rbp-58h]
  _BYTE v54[80]; // [rsp+F0h] [rbp-50h] BYREF

  v2 = a1;
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(char ***)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 8) = sub_22077B0(96LL * (unsigned int)v5);
  if ( !v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    if ( !byte_4FAF3A0 && (unsigned int)sub_2207590(&byte_4FAF3A0) )
    {
      qword_4FAF3D0 = 0;
      qword_4FAF3C0 = (__int64)&qword_4FAF3D0;
      qword_4FAF3F0 = (__int64)&unk_4FAF400;
      qword_4FAF3F8 = 0x400000000LL;
      qword_4FAF3C8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF3C0, &qword_4A427C0);
      sub_2207640(&byte_4FAF3A0);
    }
    v49 = v51;
    v50 = 0x400000000LL;
    if ( (_DWORD)qword_4FAF3C8 )
      sub_192DAF0((__int64)&v49, (__int64)&qword_4FAF3C0, v6, v7, v8, v9);
    v52 = v54;
    v53 = 0x400000000LL;
    if ( (_DWORD)qword_4FAF3F8 )
    {
      sub_192DA10((__int64)&v52, (__int64)&qword_4FAF3F0, v6, v7, v8, v9);
      v37 = *(_QWORD *)(a1 + 8);
      v38 = v37 + 96LL * *(unsigned int *)(a1 + 24);
      if ( v37 != v38 )
        goto LABEL_96;
    }
    else
    {
      v37 = *(_QWORD *)(a1 + 8);
      v38 = v37 + 96LL * *(unsigned int *)(a1 + 24);
      if ( v37 == v38 )
      {
LABEL_102:
        if ( v49 != v51 )
          _libc_free((unsigned __int64)v49);
        return;
      }
      do
      {
LABEL_96:
        while ( 1 )
        {
          if ( v37 )
          {
            *(_DWORD *)(v37 + 8) = 0;
            *(_QWORD *)v37 = v37 + 16;
            *(_DWORD *)(v37 + 12) = 4;
            if ( (_DWORD)v50 )
              sub_192DAF0(v37, (__int64)&v49, v6, v7, v8, v9);
            *(_DWORD *)(v37 + 56) = 0;
            *(_QWORD *)(v37 + 48) = v37 + 64;
            *(_DWORD *)(v37 + 60) = 4;
            if ( (_DWORD)v53 )
              break;
          }
          v37 += 96;
          if ( v37 == v38 )
            goto LABEL_100;
        }
        v39 = v37 + 48;
        v37 += 96;
        sub_192DA10(v39, (__int64)&v52, v6, v7, v8, v9);
      }
      while ( v37 != v38 );
    }
LABEL_100:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
    goto LABEL_102;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v10 = &v4[12 * v3];
  if ( !byte_4FAF3A0 && (unsigned int)sub_2207590(&byte_4FAF3A0) )
  {
    qword_4FAF3D0 = 0;
    qword_4FAF3C0 = (__int64)&qword_4FAF3D0;
    qword_4FAF3F0 = (__int64)&unk_4FAF400;
    qword_4FAF3F8 = 0x400000000LL;
    qword_4FAF3C8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF3C0, &qword_4A427C0);
    sub_2207640(&byte_4FAF3A0);
  }
  v49 = v51;
  v50 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3C8 )
    sub_192DAF0((__int64)&v49, (__int64)&qword_4FAF3C0, v6, v7, v8, v9);
  v11 = qword_4FAF3F8;
  v52 = v54;
  v53 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3F8 )
  {
    sub_192DA10((__int64)&v52, (__int64)&qword_4FAF3F0, v6, v7, v8, qword_4FAF3F8);
    v12 = *(_QWORD *)(a1 + 8);
    v13 = 96LL * *(unsigned int *)(a1 + 24);
    v14 = v12 + v13;
    if ( v12 == v12 + v13 )
    {
LABEL_17:
      if ( v52 != v54 )
        _libc_free((unsigned __int64)v52);
      goto LABEL_19;
    }
    do
    {
LABEL_13:
      if ( v12 )
      {
        *(_DWORD *)(v12 + 8) = 0;
        *(_QWORD *)v12 = v12 + 16;
        *(_DWORD *)(v12 + 12) = 4;
        v8 = v50;
        if ( (_DWORD)v50 )
          sub_192DAF0(v12, (__int64)&v49, v12 + 16, v7, v50, v11);
        v13 = v12 + 64;
        *(_DWORD *)(v12 + 56) = 0;
        *(_QWORD *)(v12 + 48) = v12 + 64;
        *(_DWORD *)(v12 + 60) = 4;
        if ( (_DWORD)v53 )
          sub_192DA10(v12 + 48, (__int64)&v52, v13, v7, v8, v11);
      }
      v12 += 96;
    }
    while ( v12 != v14 );
    v2 = a1;
    goto LABEL_17;
  }
  v12 = *(_QWORD *)(a1 + 8);
  v13 = 96LL * *(unsigned int *)(a1 + 24);
  v14 = v12 + v13;
  if ( v12 != v12 + v13 )
    goto LABEL_13;
LABEL_19:
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( !byte_4FAF3A0 && (unsigned int)sub_2207590(&byte_4FAF3A0) )
  {
    qword_4FAF3D0 = 0;
    qword_4FAF3C0 = (__int64)&qword_4FAF3D0;
    qword_4FAF3F0 = (__int64)&unk_4FAF400;
    qword_4FAF3F8 = 0x400000000LL;
    qword_4FAF3C8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF3C0, &qword_4A427C0);
    sub_2207640(&byte_4FAF3A0);
  }
  s2 = v45;
  v44 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3C8 )
    sub_192DAF0((__int64)&s2, (__int64)&qword_4FAF3C0, v13, v7, v8, v11);
  v15 = (unsigned int)qword_4FAF3F8;
  v46 = v48;
  v47 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF3F8 )
    sub_192DA10((__int64)&v46, (__int64)&qword_4FAF3F0, v13, (unsigned int)qword_4FAF3F8, v8, v11);
  if ( !byte_4FAF328 && (unsigned int)sub_2207590(&byte_4FAF328) )
  {
    qword_4FAF350 = 1;
    qword_4FAF340 = (__int64)&qword_4FAF350;
    qword_4FAF370 = (__int64)&unk_4FAF380;
    qword_4FAF378 = 0x400000000LL;
    qword_4FAF348 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_192D0E0, &qword_4FAF340, &qword_4A427C0);
    sub_2207640(&byte_4FAF328);
  }
  v16 = (unsigned int)qword_4FAF348;
  v49 = v51;
  v50 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF348 )
    sub_192DAF0((__int64)&v49, (__int64)&qword_4FAF340, (unsigned int)qword_4FAF348, v15, v8, v11);
  v52 = v54;
  v53 = 0x400000000LL;
  if ( (_DWORD)qword_4FAF378 )
  {
    sub_192DA10((__int64)&v52, (__int64)&qword_4FAF370, v16, v15, v8, v11);
    if ( v10 == v4 )
      goto LABEL_46;
  }
  else if ( v10 == v4 )
  {
    goto LABEL_48;
  }
  v40 = v4;
  v17 = (char *)(v4 + 2);
  v18 = v4;
  v41 = v2;
  do
  {
    v19 = *((unsigned int *)v18 + 2);
    if ( v19 != (unsigned int)v44 )
      goto LABEL_33;
    v27 = 8 * v19;
    if ( 8 * v19 )
    {
      v28 = *v18;
      if ( memcmp(*v18, s2, 8 * v19) )
      {
        if ( v19 != (unsigned int)v50 )
          goto LABEL_34;
        v31 = v49;
        goto LABEL_68;
      }
    }
    v29 = *((unsigned int *)v18 + 14);
    if ( v29 == (unsigned int)v47 )
    {
      v30 = 8 * v29;
      v26 = (unsigned __int64)v18[6];
      if ( !v30 || !memcmp(v18[6], v46, v30) )
        goto LABEL_40;
      if ( v19 != (unsigned int)v50 )
        goto LABEL_34;
    }
    else
    {
LABEL_33:
      if ( v19 != (unsigned int)v50 )
        goto LABEL_34;
    }
    v31 = v49;
    v28 = *v18;
    v27 = 8 * v19;
    if ( !(8 * v19) )
      goto LABEL_62;
LABEL_68:
    if ( !memcmp(v28, v31, v27) )
    {
LABEL_62:
      v32 = *((unsigned int *)v18 + 14);
      if ( v32 == (unsigned int)v53 )
      {
        v33 = 8 * v32;
        v26 = (unsigned __int64)v18[6];
        if ( !v33 || !memcmp(v18[6], v52, v33) )
          goto LABEL_40;
      }
    }
LABEL_34:
    sub_1931280(v41, (const void **)v18, &v42, v15, v8, v11);
    v24 = v42;
    if ( (char **)v42 != v18 )
    {
      v25 = *v18;
      if ( v17 == *v18 )
      {
        v20 = *((unsigned int *)v18 + 2);
        v34 = *(unsigned int *)(v42 + 8);
        v35 = *((_DWORD *)v18 + 2);
        if ( v20 <= v34 )
        {
          if ( *((_DWORD *)v18 + 2) )
            memmove(*(void **)v42, v17, 8 * v20);
        }
        else
        {
          if ( v20 > *(unsigned int *)(v42 + 12) )
          {
            *(_DWORD *)(v42 + 8) = 0;
            sub_16CD150(v24, (const void *)(v24 + 16), v20, 8, v22, v23);
            v25 = *v18;
            v20 = *((unsigned int *)v18 + 2);
            v34 = 0;
            v36 = *v18;
          }
          else
          {
            v21 = 8 * v34;
            v36 = v17;
            if ( *(_DWORD *)(v42 + 8) )
            {
              memmove(*(void **)v42, v17, v21);
              v21 = 8 * v34;
              v25 = *v18;
              v20 = *((unsigned int *)v18 + 2);
              v36 = &(*v18)[8 * v34];
              v34 *= 8LL;
            }
          }
          v20 *= 8LL;
          if ( v36 != &v25[v20] )
            memcpy((void *)(v34 + *(_QWORD *)v24), v36, v20 - v34);
        }
        *(_DWORD *)(v24 + 8) = v35;
        *((_DWORD *)v18 + 2) = 0;
      }
      else
      {
        v20 = v42 + 16;
        if ( *(_QWORD *)v42 != v42 + 16 )
        {
          _libc_free(*(_QWORD *)v42);
          v25 = *v18;
        }
        *(_QWORD *)v24 = v25;
        *(_DWORD *)(v24 + 8) = *((_DWORD *)v18 + 2);
        *(_DWORD *)(v24 + 12) = *((_DWORD *)v18 + 3);
        *v18 = v17;
        *((_DWORD *)v18 + 3) = 0;
        *((_DWORD *)v18 + 2) = 0;
      }
    }
    sub_192DBD0(v24 + 48, v18 + 6, v20, v21, v22, v23);
    ++*(_DWORD *)(v41 + 16);
    v26 = (unsigned __int64)v18[6];
LABEL_40:
    if ( (char **)v26 != v18 + 8 )
      _libc_free(v26);
    if ( v17 != *v18 )
      _libc_free((unsigned __int64)*v18);
    v18 += 12;
    v17 += 96;
  }
  while ( v10 != v18 );
  v4 = v40;
LABEL_46:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
LABEL_48:
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( s2 != v45 )
    _libc_free((unsigned __int64)s2);
  j___libc_free_0(v4);
}
