// Function: sub_1051870
// Address: 0x1051870
//
void __fastcall sub_1051870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // r8
  unsigned int v9; // r13d
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // r13d
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  void **v15; // r14
  void **v16; // r13
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // r13d
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  void **v25; // r14
  unsigned __int64 v26; // r8
  void **v27; // rcx
  unsigned __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned int i; // r14d
  __int64 v32; // rcx
  int v33; // r13d
  __int64 v34; // r8
  size_t v35; // rdx
  __int64 v36; // r9
  __int64 v37; // r13
  int v38; // r14d
  unsigned int v39; // r13d
  size_t v40; // rdx
  int v41; // r14d
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // rdx
  void **v45; // r14
  void **v46; // r13
  __int64 v47; // r14
  __int64 v48; // r12
  void **v49; // r13
  void **v50; // r12
  void **v51; // r12
  __int64 v52; // r8
  char *v53; // r13
  __int64 v54; // rdi
  char *v55; // r13
  __int64 v56; // rdi
  char *v57; // r13
  void **v58; // [rsp+0h] [rbp-90h]
  unsigned int v59; // [rsp+8h] [rbp-88h]
  __int64 v60; // [rsp+8h] [rbp-88h]
  unsigned __int64 v61; // [rsp+8h] [rbp-88h]
  __int64 v62; // [rsp+8h] [rbp-88h]
  int v63; // [rsp+8h] [rbp-88h]
  int v64; // [rsp+8h] [rbp-88h]
  unsigned int v65; // [rsp+8h] [rbp-88h]
  void *v66; // [rsp+10h] [rbp-80h] BYREF
  __int64 v67; // [rsp+18h] [rbp-78h]
  _BYTE s[48]; // [rsp+20h] [rbp-70h] BYREF
  int v69; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 1360) )
  {
    v7 = *(_DWORD *)(a1 + 8);
    if ( v7 )
    {
      if ( v7 != 1 )
        return;
      v8 = *(unsigned int *)(a1 + 56);
      v66 = s;
      v67 = 0x600000000LL;
      v9 = (unsigned int)(v8 + 63) >> 6;
      if ( v9 > 6 )
      {
        v65 = v8;
        sub_C8D5F0((__int64)&v66, s, v9, 8u, v8, a6);
        memset(v66, 0, 8LL * v9);
        LODWORD(v67) = v9;
        v8 = v65;
      }
      else
      {
        if ( v9 && 8LL * v9 )
        {
          v59 = v8;
          memset(s, 0, 8LL * v9);
          v8 = v59;
        }
        LODWORD(v67) = v9;
      }
      v10 = *(unsigned int *)(a1 + 624);
      v11 = *(unsigned int *)(a1 + 672);
      v69 = v8;
      v12 = v10;
      if ( v10 == v11 )
        goto LABEL_19;
      v13 = *(_QWORD *)(a1 + 664);
      v14 = 9 * v11;
      v15 = (void **)(v13 + 72 * v11);
      if ( v10 >= v11 )
      {
        v16 = &v66;
        v60 = v10 - v11;
        if ( v10 > *(unsigned int *)(a1 + 676) )
        {
          v54 = a1 + 664;
          if ( v13 > (unsigned __int64)&v66 || v15 <= &v66 )
          {
            sub_1051770(v54, v10, v14, v13, v8, a6);
            v16 = &v66;
            v13 = *(_QWORD *)(a1 + 664);
            v11 = *(unsigned int *)(a1 + 672);
          }
          else
          {
            v55 = (char *)&v66 - v13;
            sub_1051770(v54, v10, v14, v13, v8, a6);
            v13 = *(_QWORD *)(a1 + 664);
            v11 = *(unsigned int *)(a1 + 672);
            v16 = (void **)&v55[v13];
          }
        }
        v17 = v60;
        v18 = v13 + 72 * v11;
        do
        {
          if ( v18 )
          {
            *(_DWORD *)(v18 + 8) = 0;
            *(_QWORD *)v18 = v18 + 16;
            *(_DWORD *)(v18 + 12) = 6;
            v19 = *((unsigned int *)v16 + 2);
            if ( (_DWORD)v19 )
            {
              v10 = (unsigned __int64)v16;
              sub_104D010(v18, (__int64)v16, v14, v19, v8, a6);
            }
            *(_DWORD *)(v18 + 64) = *((_DWORD *)v16 + 16);
          }
          v18 += 72;
          --v17;
        }
        while ( v17 );
LABEL_18:
        *(_DWORD *)(a1 + 672) += v60;
        goto LABEL_19;
      }
      v50 = (void **)(v13 + 72 * v10);
      while ( v50 != v15 )
      {
        v15 -= 9;
        if ( *v15 != v15 + 2 )
          _libc_free(*v15, v10);
      }
    }
    else
    {
      v38 = *(_DWORD *)(a1 + 56);
      v66 = s;
      v67 = 0x600000000LL;
      v39 = (unsigned int)(v38 + 63) >> 6;
      if ( v39 > 6 )
      {
        sub_C8D5F0((__int64)&v66, s, v39, 8u, a5, a6);
        memset(v66, 255, 8LL * v39);
        LODWORD(v67) = (unsigned int)(v38 + 63) >> 6;
      }
      else
      {
        if ( v39 )
        {
          v40 = 8LL * v39;
          if ( v40 )
            memset(s, 255, v40);
        }
        LODWORD(v67) = (unsigned int)(v38 + 63) >> 6;
      }
      v69 = v38;
      v41 = v38 & 0x3F;
      if ( v41 )
        *((_QWORD *)v66 + (unsigned int)v67 - 1) &= ~(-1LL << v41);
      v10 = *(unsigned int *)(a1 + 624);
      v42 = *(unsigned int *)(a1 + 672);
      v12 = *(_DWORD *)(a1 + 624);
      if ( v10 == v42 )
        goto LABEL_19;
      v43 = *(_QWORD *)(a1 + 664);
      v44 = 9 * v42;
      v45 = (void **)(v43 + 72 * v42);
      if ( v10 >= v42 )
      {
        v46 = &v66;
        v60 = v10 - v42;
        if ( v10 > *(unsigned int *)(a1 + 676) )
        {
          v56 = a1 + 664;
          if ( v43 > (unsigned __int64)&v66 || v45 <= &v66 )
          {
            sub_1051770(v56, v10, v44, v43, a5, a6);
            v46 = &v66;
            v43 = *(_QWORD *)(a1 + 664);
            v42 = *(unsigned int *)(a1 + 672);
          }
          else
          {
            v57 = (char *)&v66 - v43;
            sub_1051770(v56, v10, v44, v43, a5, a6);
            v43 = *(_QWORD *)(a1 + 664);
            v42 = *(unsigned int *)(a1 + 672);
            v46 = (void **)&v57[v43];
          }
        }
        v47 = v60;
        v48 = v43 + 72 * v42;
        do
        {
          if ( v48 )
          {
            *(_DWORD *)(v48 + 8) = 0;
            *(_QWORD *)v48 = v48 + 16;
            *(_DWORD *)(v48 + 12) = 6;
            v10 = *((unsigned int *)v46 + 2);
            if ( (_DWORD)v10 )
            {
              v10 = (unsigned __int64)v46;
              sub_104D010(v48, (__int64)v46, v44, v43, a5, a6);
            }
            *(_DWORD *)(v48 + 64) = *((_DWORD *)v46 + 16);
          }
          v48 += 72;
          --v47;
        }
        while ( v47 );
        goto LABEL_18;
      }
      v51 = (void **)(v43 + 72 * v10);
      while ( v51 != v45 )
      {
        while ( 1 )
        {
          v45 -= 9;
          if ( *v45 == v45 + 2 )
            break;
          _libc_free(*v45, v10);
          if ( v51 == v45 )
            goto LABEL_80;
        }
      }
    }
LABEL_80:
    *(_DWORD *)(a1 + 672) = v12;
LABEL_19:
    if ( v66 != s )
      _libc_free(v66, v10);
    return;
  }
  v20 = *(unsigned int *)(a1 + 56);
  v66 = s;
  v67 = 0x600000000LL;
  v21 = (unsigned int)(v20 + 63) >> 6;
  if ( v21 > 6 )
  {
    v64 = v20;
    sub_C8D5F0((__int64)&v66, s, v21, 8u, v20, a6);
    memset(v66, 0, 8LL * v21);
    LODWORD(v67) = v21;
    LODWORD(v20) = v64;
  }
  else
  {
    if ( v21 && 8LL * v21 )
    {
      v63 = v20;
      memset(s, 0, 8LL * v21);
      LODWORD(v20) = v63;
    }
    LODWORD(v67) = v21;
  }
  v22 = *(unsigned int *)(a1 + 624);
  v23 = *(unsigned int *)(a1 + 672);
  v69 = v20;
  if ( v22 != v23 )
  {
    v24 = *(_QWORD *)(a1 + 664);
    v25 = (void **)(v24 + 72 * v23);
    if ( v22 < v23 )
    {
      v49 = (void **)(v24 + 72 * v22);
      while ( v49 != v25 )
      {
        v25 -= 9;
        if ( *v25 != v25 + 2 )
          _libc_free(*v25, v22);
      }
      *(_DWORD *)(a1 + 672) = v22;
    }
    else
    {
      v26 = *(unsigned int *)(a1 + 676);
      v27 = &v66;
      v61 = v22 - v23;
      if ( v22 > v26 )
      {
        v52 = a1 + 664;
        if ( v24 > (unsigned __int64)&v66 || v25 <= &v66 )
        {
          sub_1051770(a1 + 664, v22, v23, (__int64)&v66, v52, a6);
          v24 = *(_QWORD *)(a1 + 664);
          v23 = *(unsigned int *)(a1 + 672);
          v27 = &v66;
        }
        else
        {
          v53 = (char *)&v66 - v24;
          sub_1051770(a1 + 664, v22, v23, (__int64)&v66, v52, a6);
          v24 = *(_QWORD *)(a1 + 664);
          v23 = *(unsigned int *)(a1 + 672);
          v27 = (void **)&v53[v24];
        }
      }
      v28 = v61;
      v29 = v24 + 72 * v23;
      do
      {
        if ( v29 )
        {
          *(_DWORD *)(v29 + 8) = 0;
          *(_QWORD *)v29 = v29 + 16;
          *(_DWORD *)(v29 + 12) = 6;
          v30 = *((unsigned int *)v27 + 2);
          if ( (_DWORD)v30 )
          {
            v22 = (unsigned __int64)v27;
            v58 = v27;
            sub_104D010(v29, (__int64)v27, v30, (__int64)v27, v26, a6);
            v27 = v58;
          }
          *(_DWORD *)(v29 + 64) = *((_DWORD *)v27 + 16);
        }
        v29 += 72;
        --v28;
      }
      while ( v28 );
      *(_DWORD *)(a1 + 672) += v61;
    }
  }
  if ( v66 != s )
    _libc_free(v66, v22);
  for ( i = 0; *(_DWORD *)(a1 + 624) > i; ++i )
  {
    while ( 1 )
    {
      v32 = i;
      if ( (*(_QWORD *)(*(_QWORD *)(a1 + 1256) + 8LL * (i >> 6)) & (1LL << i)) == 0 )
      {
        v33 = *(_DWORD *)(a1 + 56);
        v66 = s;
        v67 = 0x600000000LL;
        v34 = (unsigned int)(v33 + 63) >> 6;
        if ( (unsigned int)v34 > 6 )
        {
          v62 = (unsigned int)v34;
          sub_C8D5F0((__int64)&v66, s, (unsigned int)v34, 8u, v34, a6);
          memset(v66, 255, 8 * v62);
          v34 = (unsigned int)(v33 + 63) >> 6;
          LODWORD(v67) = (unsigned int)(v33 + 63) >> 6;
        }
        else
        {
          if ( (_DWORD)v34 )
          {
            v35 = 8LL * (unsigned int)v34;
            if ( v35 )
            {
              memset(s, 255, v35);
              v34 = (unsigned int)(v33 + 63) >> 6;
            }
          }
          LODWORD(v67) = v34;
        }
        v69 = v33;
        v36 = v33 & 0x3F;
        if ( (v33 & 0x3F) != 0 )
        {
          v32 = v33 & 0x3F;
          *((_QWORD *)v66 + (unsigned int)v67 - 1) &= ~(-1LL << v36);
        }
        v37 = *(_QWORD *)(a1 + 664) + 72LL * i;
        sub_104D0F0(v37, (char **)&v66, 9LL * i, v32, v34, v36);
        *(_DWORD *)(v37 + 64) = v69;
        if ( v66 != s )
          break;
      }
      if ( *(_DWORD *)(a1 + 624) <= ++i )
        goto LABEL_50;
    }
    _libc_free(v66, &v66);
  }
LABEL_50:
  sub_104E600(a1);
  sub_1050730(a1);
}
