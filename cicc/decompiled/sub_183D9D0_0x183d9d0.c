// Function: sub_183D9D0
// Address: 0x183d9d0
//
__int64 __fastcall sub_183D9D0(size_t *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  int *v6; // rdi
  size_t v7; // rdx
  size_t v8; // rbx
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  size_t v13; // r12
  int v14; // ecx
  _QWORD *v15; // rax
  int v16; // ebx
  size_t v17; // r15
  _QWORD *v18; // r13
  void *v19; // rax
  void *v20; // r8
  size_t v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int64 v26; // r12
  unsigned int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // r15
  __int64 v30; // rdx
  void *v31; // rax
  void *v32; // r12
  int v33; // r13d
  size_t v34; // rbx
  void (__fastcall *v35)(char ***, size_t, int *, int *); // r15
  size_t v36; // r9
  __int64 v37; // rcx
  void *v38; // rax
  char *v39; // rax
  unsigned __int64 v40; // r12
  __int64 v41; // rax
  char *v42; // rcx
  size_t v43; // r13
  _BYTE *v44; // rdi
  void *v45; // rax
  char *v46; // rax
  char *v47; // rax
  char *v48; // r9
  int *v49; // r8
  size_t v50; // rbx
  __int64 v51; // rax
  char *v52; // rcx
  size_t v53; // r12
  __int64 result; // rax
  char *v55; // rax
  char *v56; // rbx
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // eax
  int v61; // eax
  int v62; // eax
  size_t v63; // [rsp+0h] [rbp-170h]
  int *v64; // [rsp+8h] [rbp-168h]
  char *v65; // [rsp+10h] [rbp-160h]
  char *v66; // [rsp+10h] [rbp-160h]
  char *v67; // [rsp+18h] [rbp-158h]
  unsigned __int64 v68; // [rsp+20h] [rbp-150h]
  int *dest; // [rsp+28h] [rbp-148h]
  size_t v70; // [rsp+30h] [rbp-140h]
  int v71; // [rsp+30h] [rbp-140h]
  size_t v72; // [rsp+30h] [rbp-140h]
  int *v73; // [rsp+30h] [rbp-140h]
  size_t n; // [rsp+40h] [rbp-130h]
  int v75; // [rsp+54h] [rbp-11Ch]
  __int64 v76; // [rsp+58h] [rbp-118h]
  size_t v78; // [rsp+78h] [rbp-F8h]
  int v79; // [rsp+78h] [rbp-F8h]
  __int64 v80; // [rsp+78h] [rbp-F8h]
  int v81; // [rsp+78h] [rbp-F8h]
  void *v82; // [rsp+78h] [rbp-F8h]
  int v83; // [rsp+80h] [rbp-F0h] BYREF
  void *s2; // [rsp+88h] [rbp-E8h]
  char *v85; // [rsp+90h] [rbp-E0h]
  char *v86; // [rsp+98h] [rbp-D8h]
  int v87; // [rsp+A0h] [rbp-D0h] BYREF
  char *v88; // [rsp+A8h] [rbp-C8h]
  char *v89; // [rsp+B0h] [rbp-C0h]
  char *v90; // [rsp+B8h] [rbp-B8h]
  int v91; // [rsp+C0h] [rbp-B0h] BYREF
  void *src; // [rsp+C8h] [rbp-A8h]
  void *v93; // [rsp+D0h] [rbp-A0h]
  __int64 v94; // [rsp+D8h] [rbp-98h]
  int v95; // [rsp+E0h] [rbp-90h] BYREF
  char *v96; // [rsp+E8h] [rbp-88h]
  char *v97; // [rsp+F0h] [rbp-80h]
  char *v98; // [rsp+F8h] [rbp-78h]
  int v99; // [rsp+100h] [rbp-70h] BYREF
  int *v100; // [rsp+108h] [rbp-68h]
  char *v101; // [rsp+110h] [rbp-60h]
  size_t v102; // [rsp+118h] [rbp-58h]
  char **v103; // [rsp+120h] [rbp-50h] BYREF
  __int64 v104; // [rsp+128h] [rbp-48h]
  char *v105; // [rsp+130h] [rbp-40h] BYREF
  char *v106; // [rsp+138h] [rbp-38h]

  v2 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  v6 = &v83;
  v68 = v2;
  sub_183C910((__int64)&v83, a1, v2);
  v8 = *a1;
  v9 = *(_QWORD *)(v8 + 48);
  v75 = *(_DWORD *)(v8 + 40);
  v10 = *(_QWORD **)(v8 + 56);
  v11 = (unsigned __int64)v10 - v9;
  n = (size_t)v10 - v9;
  v67 = (char *)v10 - v9;
  if ( v10 == (_QWORD *)v9 )
  {
    dest = 0;
    v13 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_103;
    v6 = (int *)(*(_QWORD *)(v8 + 56) - v9);
    v12 = sub_22077B0(v11);
    v9 = *(_QWORD *)(v8 + 48);
    dest = (int *)v12;
    v10 = *(_QWORD **)(v8 + 56);
    n = (size_t)v10 - v9;
    v13 = (size_t)v10 - v9;
  }
  if ( v10 != (_QWORD *)v9 )
  {
    v6 = dest;
    memmove(dest, (const void *)v9, v13);
  }
  v14 = v83;
  if ( v83 == v75 )
  {
    v6 = (int *)s2;
    if ( v85 - (_BYTE *)s2 == n )
    {
      v81 = v83;
      if ( !n )
        goto LABEL_55;
      v61 = memcmp(s2, dest, n);
      v14 = v81;
      if ( !v61 )
        goto LABEL_55;
    }
  }
  v7 = *a1;
  v15 = *(_QWORD **)(*a1 + 88);
  v9 = *(_QWORD *)(*a1 + 80);
  v78 = *a1;
  v16 = *(_DWORD *)(*a1 + 72);
  v17 = (size_t)v15 - v9;
  v18 = (_QWORD *)((char *)v15 - v9);
  if ( v15 == (_QWORD *)v9 )
  {
    v20 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_103;
    v6 = (int *)((char *)v15 - v9);
    v19 = (void *)sub_22077B0(v17);
    v7 = v78;
    v14 = v83;
    v20 = v19;
    v9 = *(_QWORD *)(v78 + 80);
    v17 = *(_QWORD *)(v78 + 88) - v9;
    if ( v9 != *(_QWORD *)(v78 + 88) )
    {
      v21 = *(_QWORD *)(v78 + 88) - v9;
      v79 = v83;
      v20 = memmove(v19, (const void *)v9, v21);
      if ( v79 != v16 )
        goto LABEL_11;
      v6 = (int *)s2;
      if ( v17 != v85 - (_BYTE *)s2 )
        goto LABEL_11;
      goto LABEL_92;
    }
  }
  if ( v14 == v16 )
  {
    v6 = (int *)s2;
    if ( v85 - (_BYTE *)s2 == v17 )
    {
LABEL_92:
      if ( !v17 )
      {
        if ( !v20 )
          goto LABEL_55;
        goto LABEL_94;
      }
      v82 = v20;
      v62 = memcmp(v6, v20, v17);
      v20 = v82;
      if ( !v62 )
      {
LABEL_94:
        j_j___libc_free_0(v20, v18);
        goto LABEL_55;
      }
      goto LABEL_11;
    }
  }
  if ( !v20 )
  {
    v22 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( (unsigned int)v22 <= 0x40 )
      goto LABEL_12;
LABEL_64:
    v88 = 0;
    v89 = 0;
    v87 = v75;
    v90 = 0;
    if ( n )
    {
      if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_103;
      v55 = (char *)sub_22077B0(v13);
      v56 = &v55[v13];
      v88 = v55;
      v90 = &v55[v13];
      memcpy(v55, dest, v13);
    }
    else
    {
      v90 = (char *)v13;
      v56 = (char *)v13;
    }
    v89 = v56;
    sub_183BFC0((__int64)a1, v68, (__int64)&v87);
    if ( v88 )
      j_j___libc_free_0(v88, v90 - v88);
    goto LABEL_55;
  }
LABEL_11:
  v9 = (unsigned __int64)v18;
  v6 = (int *)v20;
  j_j___libc_free_0(v20, v18);
  v22 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (unsigned int)v22 > 0x40 )
    goto LABEL_64;
LABEL_12:
  if ( !(_DWORD)v22 )
  {
LABEL_47:
    v48 = v85;
    v49 = (int *)s2;
    v50 = v85 - (_BYTE *)s2;
    goto LABEL_48;
  }
  v80 = 0;
  v76 = 8 * v22;
  while ( 1 )
  {
    v23 = *(_QWORD *)(a2 + 40);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v24 = *(_QWORD *)(a2 - 8);
    else
      v24 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v25 = *(_QWORD *)(v80 + v24 + 24LL * *(unsigned int *)(a2 + 56) + 8);
    v103 = &v105;
    v104 = 0x1000000000LL;
    v26 = sub_157EBA0(v25);
    v9 = v26;
    sub_183D8F0((unsigned __int64)a1, v26, &v103, 1);
    v27 = sub_15F4D60(v26);
    if ( !v27 )
    {
LABEL_69:
      v6 = (int *)v103;
      if ( v103 != &v105 )
        _libc_free((unsigned __int64)v103);
      goto LABEL_46;
    }
    v28 = v27;
    v29 = 0;
    while ( 1 )
    {
      v9 = (unsigned int)v29;
      if ( v23 == sub_15F4DF0(v26, v29) )
      {
        if ( *((_BYTE *)v103 + v29) )
          break;
      }
      if ( ++v29 == v28 )
        goto LABEL_69;
    }
    if ( v103 != &v105 )
      _libc_free((unsigned __int64)v103);
    v30 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v9 = (unsigned __int64)a1;
    v6 = &v91;
    sub_183C910((__int64)&v91, a1, *(_QWORD *)(v30 + 3 * v80) & 0xFFFFFFFFFFFFFFF9LL);
    v31 = v93;
    v32 = src;
    v33 = v83;
    v7 = (_BYTE *)v93 - (_BYTE *)src;
    if ( v91 == v83 )
    {
      v7 = (_BYTE *)v93 - (_BYTE *)src;
      v50 = v85 - (_BYTE *)s2;
      if ( (_BYTE *)v93 - (_BYTE *)src == v85 - (_BYTE *)s2 )
        break;
    }
    v34 = *a1;
    v9 = *(_QWORD *)*a1;
    v99 = v91;
    v100 = 0;
    v101 = 0;
    v35 = *(void (__fastcall **)(char ***, size_t, int *, int *))(v9 + 40);
    v102 = 0;
    if ( v7 )
      goto LABEL_77;
    v36 = 0;
    v37 = 0;
LABEL_28:
    v7 += v37;
    v100 = (int *)v37;
    v101 = (char *)v37;
    v102 = v7;
    if ( v32 != v31 )
    {
      v6 = (int *)v37;
      v70 = v36;
      v38 = memmove((void *)v37, v32, v36);
      v36 = v70;
      v37 = (__int64)v38;
    }
    v39 = v85;
    v9 = (unsigned __int64)s2;
    v95 = v33;
    v101 = (char *)(v36 + v37);
    v97 = 0;
    v96 = 0;
    v98 = 0;
    v40 = v85 - (_BYTE *)s2;
    if ( v85 == s2 )
    {
      v43 = 0;
      v42 = 0;
    }
    else
    {
      if ( v40 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_103;
      v41 = sub_22077B0(v85 - (_BYTE *)s2);
      v9 = (unsigned __int64)s2;
      v42 = (char *)v41;
      v39 = v85;
      v43 = v85 - (_BYTE *)s2;
    }
    v96 = v42;
    v97 = v42;
    v98 = &v42[v40];
    if ( v39 != (char *)v9 )
      v42 = (char *)memmove(v42, (const void *)v9, v43);
    v97 = &v42[v43];
    v35(&v103, v34, &v95, &v99);
    v44 = s2;
    v9 = (unsigned __int64)v86;
    v83 = (int)v103;
    v45 = (void *)v104;
    v104 = 0;
    s2 = v45;
    v46 = v105;
    v105 = 0;
    v85 = v46;
    v47 = v106;
    v106 = 0;
    v86 = v47;
    if ( v44 )
    {
      j_j___libc_free_0(v44, v9 - (_QWORD)v44);
      v9 = (unsigned __int64)&v106[-v104];
      if ( v104 )
        j_j___libc_free_0(v104, v9);
    }
    if ( v96 )
    {
      v9 = v98 - v96;
      j_j___libc_free_0(v96, v98 - v96);
    }
    v6 = v100;
    if ( v100 )
    {
      v9 = v102 - (_QWORD)v100;
      j_j___libc_free_0(v100, v102 - (_QWORD)v100);
    }
    v33 = v83;
    v32 = src;
LABEL_43:
    if ( v75 == v33 )
    {
      v48 = v85;
      v49 = (int *)s2;
      v50 = v85 - (_BYTE *)s2;
      if ( v85 - (_BYTE *)s2 == n )
        goto LABEL_81;
    }
LABEL_44:
    if ( v32 )
      goto LABEL_45;
LABEL_46:
    v80 += 8;
    if ( v76 == v80 )
      goto LABEL_47;
  }
  v65 = v85;
  v71 = v91;
  if ( !v7 )
    goto LABEL_43;
  v9 = (unsigned __int64)s2;
  v6 = (int *)src;
  v63 = (_BYTE *)v93 - (_BYTE *)src;
  v64 = (int *)s2;
  v57 = memcmp(src, s2, v7);
  v49 = v64;
  v7 = v63;
  v48 = v65;
  if ( v57 )
  {
    v34 = *a1;
    v58 = *(_QWORD *)*a1;
    v99 = v71;
    v100 = 0;
    v101 = 0;
    v35 = *(void (__fastcall **)(char ***, size_t, int *, int *))(v58 + 40);
    v102 = 0;
LABEL_77:
    if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_103;
    v6 = (int *)v7;
    v72 = v7;
    v59 = sub_22077B0(v7);
    v32 = src;
    v33 = v83;
    v37 = v59;
    v31 = v93;
    v7 = v72;
    v36 = (_BYTE *)v93 - (_BYTE *)src;
    goto LABEL_28;
  }
  if ( v75 != v71 || v50 != n )
  {
LABEL_45:
    v6 = (int *)v32;
    v9 = v94 - (_QWORD)v32;
    j_j___libc_free_0(v32, v94 - (_QWORD)v32);
    goto LABEL_46;
  }
LABEL_81:
  if ( v50 )
  {
    v9 = (unsigned __int64)dest;
    v6 = v49;
    v66 = v48;
    v73 = v49;
    v60 = memcmp(v49, dest, v50);
    v49 = v73;
    v48 = v66;
    if ( v60 )
      goto LABEL_44;
  }
  if ( v32 )
  {
    v6 = (int *)v32;
    v9 = v94 - (_QWORD)v32;
    j_j___libc_free_0(v32, v94 - (_QWORD)v32);
    v48 = v85;
    v49 = (int *)s2;
    v50 = v85 - (_BYTE *)s2;
  }
LABEL_48:
  v104 = 0;
  v105 = 0;
  LODWORD(v103) = v83;
  v106 = 0;
  if ( v50 )
  {
    if ( v50 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v51 = sub_22077B0(v50);
      v48 = v85;
      v49 = (int *)s2;
      v52 = (char *)v51;
      v53 = v85 - (_BYTE *)s2;
      goto LABEL_51;
    }
LABEL_103:
    sub_4261EA(v6, v9, v7);
  }
  v53 = 0;
  v52 = 0;
LABEL_51:
  v104 = (__int64)v52;
  v105 = v52;
  v106 = &v52[v50];
  if ( v48 != (char *)v49 )
    v52 = (char *)memmove(v52, v49, v53);
  v105 = &v52[v53];
  sub_183BFC0((__int64)a1, v68, (__int64)&v103);
  if ( v104 )
    j_j___libc_free_0(v104, &v106[-v104]);
LABEL_55:
  result = (__int64)dest;
  if ( dest )
    result = j_j___libc_free_0(dest, v67);
  if ( s2 )
    return j_j___libc_free_0(s2, v86 - (_BYTE *)s2);
  return result;
}
