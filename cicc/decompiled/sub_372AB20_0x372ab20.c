// Function: sub_372AB20
// Address: 0x372ab20
//
void __fastcall sub_372AB20(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v7; // ebx
  __int64 v8; // r11
  __int64 v9; // r12
  unsigned __int8 v10; // al
  __int64 v11; // r13
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // r12
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // r13
  _QWORD *v22; // r14
  __int64 v23; // rcx
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rbx
  __int64 *v26; // r14
  __int64 *v27; // r13
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // rdx
  __int64 v32; // r8
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdx
  _QWORD *v35; // r9
  _QWORD *v36; // rax
  _QWORD *v37; // rsi
  __int64 v38; // rbx
  unsigned __int8 *v39; // r14
  __int64 v40; // r8
  unsigned int v41; // eax
  unsigned int v42; // r13d
  unsigned __int64 v43; // rbx
  unsigned __int64 v44; // r12
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rsi
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // r12
  size_t v51; // r12
  char *v52; // rbx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  char *v55; // rcx
  __int64 v56; // rsi
  __int64 v57; // rdx
  _QWORD *v58; // rsi
  __int64 v59; // rax
  __int64 j; // rcx
  __int64 v61; // rdx
  char *v62; // r12
  unsigned int v63; // eax
  __int64 v64; // rdx
  __int64 v65; // rdi
  unsigned __int8 v66; // al
  _QWORD *v67; // r13
  __int64 v69; // [rsp+28h] [rbp-F8h]
  __int64 v70; // [rsp+38h] [rbp-E8h]
  __int64 *v71; // [rsp+40h] [rbp-E0h]
  __int64 v72; // [rsp+48h] [rbp-D8h]
  _QWORD *v73; // [rsp+50h] [rbp-D0h]
  __int64 v74; // [rsp+58h] [rbp-C8h]
  __int64 v75; // [rsp+60h] [rbp-C0h]
  _QWORD *i; // [rsp+68h] [rbp-B8h]
  __int64 v77; // [rsp+68h] [rbp-B8h]
  __int64 v78; // [rsp+68h] [rbp-B8h]
  void *s; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+78h] [rbp-A8h]
  _BYTE v81[16]; // [rsp+80h] [rbp-A0h] BYREF
  void *base; // [rsp+90h] [rbp-90h] BYREF
  __int64 v83; // [rsp+98h] [rbp-88h]
  _BYTE v84[32]; // [rsp+A0h] [rbp-80h] BYREF
  void *v85; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v86; // [rsp+C8h] [rbp-58h]
  _BYTE v87[80]; // [rsp+D0h] [rbp-50h] BYREF

  base = v84;
  v83 = 0x400000000LL;
  v80 = 0x400000000LL;
  v86 = 0x400000000LL;
  v4 = *(unsigned int *)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 32);
  s = v81;
  v85 = v87;
  v69 = v5 + 96 * v4;
  if ( v5 == v69 )
    goto LABEL_47;
  v75 = v5;
  do
  {
    v7 = *(_DWORD *)(v75 + 24);
    if ( !v7 )
      goto LABEL_42;
    v8 = *(_QWORD *)v75;
    v9 = *(_QWORD *)(v75 + 8);
    v10 = *(_BYTE *)(*(_QWORD *)v75 - 16LL);
    v11 = *(_QWORD *)v75 - 16LL;
    if ( v9 )
    {
      if ( (v10 & 2) != 0 )
        v12 = *(_QWORD **)(v8 - 32);
      else
        v12 = (_QWORD *)(v11 - 8LL * ((v10 >> 2) & 0xF));
      v13 = a3[9];
      v14 = v9 + 31LL * *v12;
      v15 = *(_QWORD *)(a3[8] + 8 * (v14 % v13));
      if ( !v15 )
        goto LABEL_42;
      v16 = *(_QWORD **)v15;
      v17 = *(_QWORD *)(*(_QWORD *)v15 + 208LL);
      while ( v14 != v17 || *v12 != v16[1] || v9 != v16[2] )
      {
        if ( !*v16 )
          goto LABEL_42;
        v17 = *(_QWORD *)(*v16 + 208LL);
        v15 = (__int64)v16;
        if ( (v9 + 31LL * *v12) % v13 != v17 % v13 )
          goto LABEL_42;
        v16 = (_QWORD *)*v16;
      }
      v18 = *(_QWORD *)v15 + 24LL;
      if ( !*(_QWORD *)v15 )
        goto LABEL_42;
    }
    else
    {
      if ( (v10 & 2) != 0 )
        v31 = *(__int64 **)(v8 - 32);
      else
        v31 = (__int64 *)(v11 - 8LL * ((v10 >> 2) & 0xF));
      v32 = *v31;
      v33 = a3[2];
      v34 = *v31 % v33;
      v35 = *(_QWORD **)(a3[1] + 8 * v34);
      if ( !v35 )
        goto LABEL_42;
      v36 = (_QWORD *)*v35;
      if ( v32 != *(_QWORD *)(*v35 + 8LL) )
      {
        do
        {
          v37 = (_QWORD *)*v36;
          if ( !*v36 )
            goto LABEL_42;
          v35 = v36;
          if ( v34 != v37[1] % v33 )
            goto LABEL_42;
          v36 = (_QWORD *)*v36;
        }
        while ( v32 != v37[1] );
      }
      v38 = *v35;
      if ( !*v35 )
        goto LABEL_42;
      v39 = *(unsigned __int8 **)(v38 + 24);
      v77 = *(_QWORD *)v75;
      v18 = v38 + 16;
      if ( v39 == sub_AF34D0(v39) )
      {
        v66 = *(_BYTE *)(v77 - 16);
        v67 = (v66 & 2) != 0 ? *(_QWORD **)(v77 - 32) : (_QWORD *)(v11 - 8LL * ((v66 >> 2) & 0xF));
        if ( *(_QWORD *)(v38 + 24) == *v67 )
          goto LABEL_42;
      }
      v7 = *(_DWORD *)(v75 + 24);
    }
    LODWORD(v83) = 0;
    if ( HIDWORD(v80) < v7 )
    {
      LODWORD(v80) = 0;
      sub_C8D5F0((__int64)&s, v81, v7, 4u, v14, v15);
      memset(s, 0, 4LL * v7);
      LODWORD(v80) = v7;
    }
    else
    {
      v19 = (unsigned int)v80;
      v20 = (unsigned int)v80;
      if ( v7 <= (unsigned __int64)(unsigned int)v80 )
        v20 = v7;
      if ( v20 )
      {
        memset(s, 0, 4 * v20);
        v19 = (unsigned int)v80;
      }
      if ( v7 > v19 )
      {
        v21 = v7 - v19;
        if ( v21 )
        {
          if ( 4 * v21 )
            memset((char *)s + 4 * v19, 0, 4 * v21);
        }
      }
      LODWORD(v80) = v7;
    }
    v74 = 0;
    v71 = *(__int64 **)(v18 + 80);
    v70 = *(unsigned int *)(v18 + 88);
    v22 = *(_QWORD **)(v75 + 16);
    v73 = &v22[2 * *(unsigned int *)(v75 + 24)];
    if ( v22 != v73 )
    {
      for ( i = *(_QWORD **)(v75 + 16); v73 != i; i += 2 )
      {
        if ( (*i & 4) == 0 )
        {
          v23 = i[1];
          v72 = v23;
          if ( v23 == -1 )
          {
            v24 = *i & 0xFFFFFFFFFFFFFFF8LL;
            v25 = 0;
            if ( *((int *)s + v74) <= 0 )
              goto LABEL_32;
          }
          else
          {
            ++*((_DWORD *)s + v23);
            if ( *((int *)s + v74) <= 0 )
            {
              v24 = *i & 0xFFFFFFFFFFFFFFF8LL;
              v25 = *(_QWORD *)(*(_QWORD *)(v75 + 16) + 16 * v23) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_32:
              v26 = v71;
              v27 = &v71[2 * v70];
              if ( v27 == v71 )
              {
LABEL_52:
                v29 = (unsigned int)v83;
                v30 = (unsigned int)v83 + 1LL;
                if ( v30 > HIDWORD(v83) )
                {
                  sub_C8D5F0((__int64)&base, v84, v30, 8u, v14, v15);
                  v29 = (unsigned int)v83;
                }
                *((_QWORD *)base + v29) = v74;
                LODWORD(v83) = v83 + 1;
                if ( v72 != -1 )
                  --*((_DWORD *)s + v72);
              }
              else
              {
                while ( 1 )
                {
                  if ( v25 )
                  {
                    if ( sub_372AA30(a4, v25, *v26) )
                      goto LABEL_52;
                    if ( !sub_372AA30(a4, v26[1], v25) )
                      break;
                  }
                  if ( sub_372AA30(a4, v24, v26[1]) )
                    break;
                  v26 += 2;
                  if ( v27 == v26 )
                    goto LABEL_52;
                }
                v71 = v26;
                v70 = ((char *)v27 - (char *)v26) >> 4;
              }
            }
          }
        }
        ++v74;
      }
    }
    v28 = (unsigned int)v83;
    if ( (_DWORD)v83 )
    {
      v40 = v75;
      v41 = *(_DWORD *)(v75 + 24);
      v42 = v41;
      if ( v41 )
      {
        v43 = 0;
        do
        {
          while ( *((int *)s + v43) > 0 || (*(_BYTE *)(*(_QWORD *)(v40 + 16) + 16 * v43) & 4) == 0 )
          {
            ++v43;
            v44 = v41;
            v42 = v41;
            if ( v43 >= v41 )
              goto LABEL_75;
          }
          v45 = (unsigned int)v28;
          v46 = (unsigned int)v28 + 1LL;
          if ( v46 > HIDWORD(v83) )
          {
            v78 = v40;
            sub_C8D5F0((__int64)&base, v84, v46, 8u, v40, v15);
            v45 = (unsigned int)v83;
            v40 = v78;
          }
          *((_QWORD *)base + v45) = v43++;
          v28 = (unsigned int)(v83 + 1);
          LODWORD(v83) = v83 + 1;
          v41 = *(_DWORD *)(v40 + 24);
          v44 = v41;
          v42 = v41;
        }
        while ( v43 < v41 );
LABEL_75:
        v47 = 8 * v28;
        if ( v28 <= 1 )
        {
          if ( v44 <= HIDWORD(v86) )
            goto LABEL_77;
LABEL_105:
          LODWORD(v86) = 0;
          sub_C8D5F0((__int64)&v85, v87, v44, 8u, v40, v15);
          memset(v85, 0, 8 * v44);
          goto LABEL_85;
        }
      }
      else
      {
        v47 = 8LL * (unsigned int)v83;
        if ( (unsigned int)v83 == 1 )
          goto LABEL_85;
      }
      qsort(base, v47 >> 3, 8u, (__compar_fn_t)sub_A15280);
      v44 = *(unsigned int *)(v75 + 24);
      v42 = *(_DWORD *)(v75 + 24);
      if ( v44 > HIDWORD(v86) )
        goto LABEL_105;
LABEL_77:
      v48 = (unsigned int)v86;
      v49 = (unsigned int)v86;
      if ( v44 <= (unsigned int)v86 )
        v49 = v44;
      if ( v49 )
      {
        memset(v85, 0, 8 * v49);
        v48 = (unsigned int)v86;
      }
      if ( v44 > v48 )
      {
        v50 = v44 - v48;
        if ( v50 )
        {
          v51 = 8 * v50;
          if ( v51 )
            memset((char *)v85 + 8 * v48, 0, v51);
        }
      }
LABEL_85:
      v52 = (char *)base;
      LODWORD(v86) = v42;
      v53 = *(_QWORD *)base;
      v54 = *(unsigned int *)(v75 + 24);
      if ( *(_QWORD *)base < v54 )
      {
        v55 = (char *)base;
        v56 = 0;
        while ( 1 )
        {
          if ( v52 != &v55[8 * (unsigned int)v83] && *(_QWORD *)v52 == v53 )
          {
            v52 += 8;
            ++v56;
          }
          *((_QWORD *)v85 + v53) = v56;
          v54 = *(unsigned int *)(v75 + 24);
          if ( ++v53 >= v54 )
            break;
          v55 = (char *)base;
        }
        v52 = (char *)base;
      }
      v57 = 16 * v54;
      v58 = v85;
      v59 = *(_QWORD *)(v75 + 16);
      for ( j = v59 + v57; v59 != j; v59 += 16 )
      {
        v61 = *(_QWORD *)(v59 + 8);
        if ( v61 != -1 )
          *(_QWORD *)(v59 + 8) = v61 - v58[v61];
      }
      v62 = &v52[8 * (unsigned int)v83];
      if ( v62 != v52 )
      {
        v63 = *(_DWORD *)(v75 + 24);
        do
        {
          v64 = *(_QWORD *)(v75 + 16) + 16LL * v63;
          v65 = *(_QWORD *)(v75 + 16) + 16LL * *((_QWORD *)v62 - 1);
          if ( v64 != v65 + 16 )
          {
            memmove((void *)v65, (const void *)(v65 + 16), v64 - (v65 + 16));
            v63 = *(_DWORD *)(v75 + 24);
          }
          --v63;
          v62 -= 8;
          *(_DWORD *)(v75 + 24) = v63;
        }
        while ( v62 != v52 );
      }
    }
LABEL_42:
    v75 += 96;
  }
  while ( v69 != v75 );
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
  if ( s != v81 )
    _libc_free((unsigned __int64)s);
LABEL_47:
  if ( base != v84 )
    _libc_free((unsigned __int64)base);
}
