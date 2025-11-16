// Function: sub_25A8730
// Address: 0x25a8730
//
void __fastcall sub_25A8730(size_t *a1, __int64 a2)
{
  int *v3; // rdi
  size_t v4; // rdx
  size_t v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  size_t v10; // r12
  int v11; // ecx
  size_t v12; // r15
  int v13; // ebx
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  void *v17; // r8
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  int v22; // eax
  __int64 v23; // rbx
  unsigned int v24; // eax
  int *v25; // r14
  __int64 v26; // r12
  __int64 v27; // r15
  void *v28; // rax
  void *v29; // r12
  int v30; // r14d
  int v31; // r13d
  size_t v32; // rbx
  __int64 v33; // rcx
  void (__fastcall *v34)(int **, size_t, int *, int *); // r15
  size_t v35; // r14
  unsigned __int64 v36; // rcx
  int *v37; // rax
  unsigned __int64 v38; // r12
  __int64 v39; // rax
  char *v40; // rcx
  size_t v41; // r13
  void *v42; // rdi
  void *v43; // rax
  int *v44; // rax
  __int64 v45; // rax
  int *v46; // rcx
  int *v47; // r8
  size_t v48; // rbx
  __int64 v49; // rax
  char *v50; // rdi
  size_t v51; // r12
  unsigned __int64 v52; // rdi
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // eax
  int v57; // eax
  char *v58; // rax
  char *v59; // rbx
  int v60; // eax
  size_t v61; // [rsp+0h] [rbp-170h]
  int *v62; // [rsp+8h] [rbp-168h]
  unsigned __int64 v63; // [rsp+18h] [rbp-158h]
  int *dest; // [rsp+20h] [rbp-150h]
  size_t n; // [rsp+28h] [rbp-148h]
  int *v66; // [rsp+30h] [rbp-140h]
  size_t v67; // [rsp+30h] [rbp-140h]
  int *v68; // [rsp+30h] [rbp-140h]
  int v69; // [rsp+3Ch] [rbp-134h]
  __int64 v70; // [rsp+40h] [rbp-130h]
  int *v71; // [rsp+48h] [rbp-128h]
  int v74; // [rsp+68h] [rbp-108h]
  __int64 v75; // [rsp+68h] [rbp-108h]
  int v76; // [rsp+68h] [rbp-108h]
  void *v77; // [rsp+68h] [rbp-108h]
  int v78; // [rsp+70h] [rbp-100h] BYREF
  void *s2; // [rsp+78h] [rbp-F8h]
  int *v80; // [rsp+80h] [rbp-F0h]
  __int64 v81; // [rsp+88h] [rbp-E8h]
  int v82; // [rsp+90h] [rbp-E0h] BYREF
  unsigned __int64 v83; // [rsp+98h] [rbp-D8h]
  char *v84; // [rsp+A0h] [rbp-D0h]
  char *v85; // [rsp+A8h] [rbp-C8h]
  int v86; // [rsp+B0h] [rbp-C0h] BYREF
  void *src; // [rsp+B8h] [rbp-B8h]
  void *v88; // [rsp+C0h] [rbp-B0h]
  __int64 v89; // [rsp+C8h] [rbp-A8h]
  int v90; // [rsp+D0h] [rbp-A0h] BYREF
  _BYTE *v91; // [rsp+D8h] [rbp-98h]
  char *v92; // [rsp+E0h] [rbp-90h]
  char *v93; // [rsp+E8h] [rbp-88h]
  int v94; // [rsp+F0h] [rbp-80h] BYREF
  int *v95; // [rsp+F8h] [rbp-78h]
  unsigned __int64 v96; // [rsp+100h] [rbp-70h]
  size_t v97; // [rsp+108h] [rbp-68h]
  int *v98; // [rsp+110h] [rbp-60h] BYREF
  void *v99; // [rsp+118h] [rbp-58h]
  __int64 v100; // [rsp+120h] [rbp-50h]
  _QWORD v101[9]; // [rsp+128h] [rbp-48h] BYREF

  v3 = &v78;
  v63 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  sub_25A73A0((__int64)&v78, a1, a2 & 0xFFFFFFFFFFFFFFF9LL);
  v5 = *a1;
  v6 = *(_QWORD *)(v5 + 48);
  v69 = *(_DWORD *)(v5 + 40);
  v7 = *(_QWORD *)(v5 + 56);
  v8 = v7 - v6;
  n = v7 - v6;
  if ( v7 == v6 )
  {
    dest = 0;
    v10 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_103;
    v3 = (int *)(*(_QWORD *)(v5 + 56) - v6);
    v9 = sub_22077B0(v8);
    v6 = *(_QWORD *)(v5 + 48);
    dest = (int *)v9;
    v7 = *(_QWORD *)(v5 + 56);
    v4 = v7 - v6;
    n = v7 - v6;
    v10 = v7 - v6;
  }
  if ( v7 != v6 )
  {
    v3 = dest;
    memmove(dest, (const void *)v6, v10);
  }
  v11 = v78;
  if ( v78 == v69 )
  {
    v3 = (int *)s2;
    if ( (char *)v80 - (_BYTE *)s2 == n )
    {
      v76 = v78;
      if ( !n )
        goto LABEL_56;
      v57 = memcmp(s2, dest, n);
      v11 = v76;
      if ( !v57 )
        goto LABEL_56;
    }
  }
  v12 = *a1;
  v6 = *(_QWORD *)(*a1 + 80);
  v13 = *(_DWORD *)(*a1 + 72);
  v14 = *(_QWORD *)(*a1 + 88) - v6;
  v15 = v14;
  if ( !v14 )
  {
    v17 = 0;
    if ( v6 != *(_QWORD *)(*a1 + 88) )
      goto LABEL_10;
LABEL_62:
    if ( v11 != v13 || (v3 = (int *)s2, (char *)v80 - (_BYTE *)s2 != v14) )
    {
      if ( !v17 )
        goto LABEL_12;
      goto LABEL_11;
    }
    goto LABEL_87;
  }
  if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_103;
  v3 = (int *)(*(_QWORD *)(*a1 + 88) - v6);
  v16 = sub_22077B0(v14);
  v6 = *(_QWORD *)(v12 + 80);
  v11 = v78;
  v17 = (void *)v16;
  v14 = *(_QWORD *)(v12 + 88) - v6;
  if ( v6 == *(_QWORD *)(v12 + 88) )
    goto LABEL_62;
LABEL_10:
  v74 = v11;
  v17 = memmove(v17, (const void *)v6, v14);
  if ( v74 == v13 )
  {
    v3 = (int *)s2;
    if ( v14 == (char *)v80 - (_BYTE *)s2 )
    {
LABEL_87:
      if ( !v14 )
      {
        if ( !v17 )
          goto LABEL_56;
        goto LABEL_89;
      }
      v77 = v17;
      v60 = memcmp(v3, v17, v14);
      v17 = v77;
      if ( !v60 )
      {
LABEL_89:
        j_j___libc_free_0((unsigned __int64)v17);
        goto LABEL_56;
      }
    }
  }
LABEL_11:
  v6 = v15;
  v3 = (int *)v17;
  j_j___libc_free_0((unsigned __int64)v17);
LABEL_12:
  v18 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (unsigned int)v18 > 0x40 )
  {
    v83 = 0;
    v84 = 0;
    v82 = v69;
    v85 = 0;
    if ( n )
    {
      if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_103;
      v58 = (char *)sub_22077B0(v10);
      v59 = &v58[v10];
      v83 = (unsigned __int64)v58;
      v85 = &v58[v10];
      memcpy(v58, dest, v10);
    }
    else
    {
      v85 = (char *)v10;
      v59 = (char *)v10;
    }
    v84 = v59;
    sub_25A6B00((__int64)a1, v63, (__int64)&v82);
    v52 = v83;
    if ( v83 )
      goto LABEL_55;
    goto LABEL_56;
  }
  if ( !(_DWORD)v18 )
  {
LABEL_48:
    v46 = v80;
    v47 = (int *)s2;
    v48 = (char *)v80 - (_BYTE *)s2;
    goto LABEL_49;
  }
  v75 = 0;
  v70 = 8 * v18;
  while ( 1 )
  {
    v19 = *(_QWORD *)(a2 + 40);
    v20 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v75);
    v99 = 0;
    v100 = 16;
    v98 = (int *)v101;
    v21 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v21 == v20 + 48 )
    {
      v23 = 0;
    }
    else
    {
      if ( !v21 )
        BUG();
      v22 = *(unsigned __int8 *)(v21 - 24);
      v23 = v21 - 24;
      if ( (unsigned int)(v22 - 30) >= 0xB )
        v23 = 0;
    }
    v6 = v23;
    sub_25A8650((unsigned __int64)a1, v23, &v98, 1);
    v3 = (int *)v23;
    v24 = sub_B46E30(v23);
    v25 = v98;
    if ( !v24 )
    {
LABEL_65:
      if ( v25 != (int *)v101 )
      {
        v3 = v25;
        _libc_free((unsigned __int64)v25);
      }
      goto LABEL_47;
    }
    v26 = v24;
    v27 = 0;
    while ( 1 )
    {
      v6 = (unsigned int)v27;
      v3 = (int *)v23;
      if ( v19 == sub_B46EC0(v23, v27) )
      {
        if ( *((_BYTE *)v25 + v27) )
          break;
      }
      if ( ++v27 == v26 )
        goto LABEL_65;
    }
    if ( v25 != (int *)v101 )
      _libc_free((unsigned __int64)v25);
    v3 = &v86;
    v6 = (__int64)a1;
    sub_25A73A0((__int64)&v86, a1, *(_QWORD *)(*(_QWORD *)(a2 - 8) + 4 * v75) & 0xFFFFFFFFFFFFFFF9LL);
    v28 = v88;
    v29 = src;
    v30 = v86;
    v31 = v78;
    v4 = (_BYTE *)v88 - (_BYTE *)src;
    if ( v86 == v78 )
    {
      v4 = (_BYTE *)v88 - (_BYTE *)src;
      v48 = (char *)v80 - (_BYTE *)s2;
      if ( (_BYTE *)v88 - (_BYTE *)src == (char *)v80 - (_BYTE *)s2 )
        break;
    }
    v32 = *a1;
    v33 = *(_QWORD *)*a1;
    v94 = v86;
    v95 = 0;
    v96 = 0;
    v34 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD))(v33 + 40);
    v97 = 0;
    if ( v4 )
      goto LABEL_71;
    v35 = 0;
    v36 = 0;
LABEL_29:
    v4 += v36;
    v95 = (int *)v36;
    v96 = v36;
    v97 = v4;
    if ( v28 != v29 )
    {
      v3 = (int *)v36;
      v36 = (unsigned __int64)memmove((void *)v36, v29, v35);
    }
    v37 = v80;
    v6 = (__int64)s2;
    v90 = v31;
    v96 = v35 + v36;
    v91 = 0;
    v92 = 0;
    v93 = 0;
    v38 = (char *)v80 - (_BYTE *)s2;
    if ( v80 == s2 )
    {
      v41 = 0;
      v40 = 0;
    }
    else
    {
      if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_103;
      v39 = sub_22077B0((char *)v80 - (_BYTE *)s2);
      v6 = (__int64)s2;
      v40 = (char *)v39;
      v37 = v80;
      v41 = (char *)v80 - (_BYTE *)s2;
    }
    v91 = v40;
    v92 = v40;
    v93 = &v40[v38];
    if ( (int *)v6 != v37 )
      v40 = (char *)memmove(v40, (const void *)v6, v41);
    v92 = &v40[v41];
    v34(&v98, v32, &v90, &v94);
    v42 = s2;
    v6 = v81;
    v78 = (int)v98;
    v43 = v99;
    v99 = 0;
    s2 = v43;
    v44 = (int *)v100;
    v100 = 0;
    v80 = v44;
    v45 = v101[0];
    v101[0] = 0;
    v81 = v45;
    if ( v42 )
    {
      j_j___libc_free_0((unsigned __int64)v42);
      v6 = v101[0] - (_QWORD)v99;
      if ( v99 )
        j_j___libc_free_0((unsigned __int64)v99);
    }
    if ( v91 )
    {
      v6 = v93 - v91;
      j_j___libc_free_0((unsigned __int64)v91);
    }
    v3 = v95;
    if ( v95 )
    {
      v6 = v97 - (_QWORD)v95;
      j_j___libc_free_0((unsigned __int64)v95);
    }
    v31 = v78;
    v29 = src;
LABEL_44:
    if ( v69 == v31 )
    {
      v46 = v80;
      v47 = (int *)s2;
      v48 = (char *)v80 - (_BYTE *)s2;
      if ( n == (char *)v80 - (_BYTE *)s2 )
        goto LABEL_75;
    }
LABEL_45:
    if ( v29 )
      goto LABEL_46;
LABEL_47:
    v75 += 8;
    if ( v70 == v75 )
      goto LABEL_48;
  }
  v66 = v80;
  if ( !v4 )
    goto LABEL_44;
  v6 = (__int64)s2;
  v3 = (int *)src;
  v61 = (_BYTE *)v88 - (_BYTE *)src;
  v62 = (int *)s2;
  v53 = memcmp(src, s2, v4);
  v47 = v62;
  v4 = v61;
  v46 = v66;
  if ( v53 )
  {
    v32 = *a1;
    v54 = *(_QWORD *)*a1;
    v94 = v30;
    v95 = 0;
    v96 = 0;
    v34 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD))(v54 + 40);
    v97 = 0;
LABEL_71:
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_103;
    v3 = (int *)v4;
    v67 = v4;
    v55 = sub_22077B0(v4);
    v29 = src;
    v31 = v78;
    v36 = v55;
    v28 = v88;
    v4 = v67;
    v35 = (_BYTE *)v88 - (_BYTE *)src;
    goto LABEL_29;
  }
  if ( v69 != v30 || n != v48 )
  {
LABEL_46:
    v3 = (int *)v29;
    v6 = v89 - (_QWORD)v29;
    j_j___libc_free_0((unsigned __int64)v29);
    goto LABEL_47;
  }
LABEL_75:
  if ( v48 )
  {
    v6 = (__int64)dest;
    v3 = v47;
    v68 = v46;
    v71 = v47;
    v56 = memcmp(v47, dest, v48);
    v47 = v71;
    v46 = v68;
    if ( v56 )
      goto LABEL_45;
  }
  if ( v29 )
  {
    v3 = (int *)v29;
    v6 = v89 - (_QWORD)v29;
    j_j___libc_free_0((unsigned __int64)v29);
    v46 = v80;
    v47 = (int *)s2;
    v48 = (char *)v80 - (_BYTE *)s2;
  }
LABEL_49:
  v99 = 0;
  v100 = 0;
  LODWORD(v98) = v78;
  v101[0] = 0;
  if ( v48 )
  {
    if ( v48 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v49 = sub_22077B0(v48);
      v46 = v80;
      v47 = (int *)s2;
      v50 = (char *)v49;
      v51 = (char *)v80 - (_BYTE *)s2;
      goto LABEL_52;
    }
LABEL_103:
    sub_4261EA(v3, v6, v4);
  }
  v51 = 0;
  v50 = 0;
LABEL_52:
  v99 = v50;
  v100 = (__int64)v50;
  v101[0] = &v50[v48];
  if ( v46 != v47 )
    v50 = (char *)memmove(v50, v47, v51);
  v100 = (__int64)&v50[v51];
  sub_25A6B00((__int64)a1, v63, (__int64)&v98);
  v52 = (unsigned __int64)v99;
  if ( v99 )
LABEL_55:
    j_j___libc_free_0(v52);
LABEL_56:
  if ( dest )
    j_j___libc_free_0((unsigned __int64)dest);
  if ( s2 )
    j_j___libc_free_0((unsigned __int64)s2);
}
