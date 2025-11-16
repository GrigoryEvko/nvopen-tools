// Function: sub_2B373F0
// Address: 0x2b373f0
//
void __fastcall sub_2B373F0(__int64 a1, __int64 a2, __int64 a3, char *a4, __int64 a5, int a6, int a7)
{
  __int64 v7; // r15
  unsigned int v12; // eax
  __int64 *v13; // rsi
  const void *v14; // r10
  __int64 v15; // r8
  signed __int64 v16; // rax
  int v17; // edx
  unsigned __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rax
  _DWORD *v22; // rdx
  signed __int64 v23; // rax
  __int64 v24; // r9
  int v25; // edx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // r8
  __int64 v31; // rax
  int v32; // edx
  signed __int64 v33; // rax
  int v34; // edx
  unsigned __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rax
  _DWORD *v39; // rdx
  signed __int64 v40; // rax
  int v41; // edx
  unsigned __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // rax
  _DWORD *v46; // rdx
  unsigned int v47; // eax
  unsigned __int64 v48; // rcx
  __int64 v49; // rdx
  unsigned int v50; // edi
  __int64 v51; // r8
  char *v52; // r10
  __int64 v53; // rax
  int v54; // edx
  signed __int64 v55; // rax
  int v56; // edx
  unsigned __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rax
  _DWORD *v61; // rdx
  unsigned int v62; // r9d
  unsigned int v63; // r15d
  __int64 v64; // rax
  unsigned int v65; // ecx
  bool v66; // cc
  __int64 v67; // [rsp+18h] [rbp-48h] BYREF
  __int64 v68; // [rsp+20h] [rbp-40h] BYREF
  void *dest[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a5;
  v12 = *(_DWORD *)(a1 + 88);
  v13 = *(__int64 **)(a1 + 80);
  if ( *(_BYTE *)(a1 + 200) )
  {
    v14 = *(const void **)(a1 + 16);
    if ( v12 == 2 )
    {
      if ( a2 == (*v13 & 0xFFFFFFFFFFFFFFF8LL) && (a3 == (v13[1] & 0xFFFFFFFFFFFFFFF8LL) || !a3) )
        goto LABEL_91;
      v15 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( !a3 && a2 == (*v13 & 0xFFFFFFFFFFFFFFF8LL) )
      {
LABEL_91:
        v62 = a7 * a6;
        dest[0] = *(void **)(a1 + 16);
        v63 = a5 - v62;
        sub_2B097A0(dest, v62);
        if ( v63 > v65 )
          v63 = v65;
        if ( 4LL * v63 )
          memmove(dest[0], &a4[4 * v64], 4LL * v63);
        return;
      }
      v15 = *(unsigned int *)(a1 + 24);
      if ( v12 == 1 )
      {
        dest[0] = 0;
        goto LABEL_6;
      }
    }
    dest[0] = (void *)v13[v12 - 1];
LABEL_6:
    v16 = sub_2B35AF0(a1, v13, (__int64)dest, v14, v15, (__int64)dest);
    if ( v17 == 1 )
      *(_DWORD *)(a1 + 128) = 1;
    if ( __OFADD__(*(_QWORD *)(a1 + 120), v16) )
    {
      v66 = v16 <= 0;
      v18 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v66 )
        v18 = 0x8000000000000000LL;
    }
    else
    {
      v18 = *(_QWORD *)(a1 + 120) + v16;
    }
    v19 = *(unsigned int *)(a1 + 24);
    v20 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 120) = v18;
    v21 = 0;
    if ( (_DWORD)v19 )
    {
      do
      {
        v22 = (_DWORD *)(v20 + 4LL * (unsigned int)v21);
        if ( *v22 != -1 )
          *v22 = v21;
        ++v21;
      }
      while ( v19 != v21 );
    }
    v13 = *(__int64 **)(a1 + 80);
    goto LABEL_15;
  }
  if ( v12 == 2 )
  {
    v40 = sub_2B35AF0(a1, v13, (__int64)(v13 + 1), *(const void **)(a1 + 16), *(unsigned int *)(a1 + 24), (__int64)dest);
    if ( v41 == 1 )
      *(_DWORD *)(a1 + 128) = 1;
    if ( __OFADD__(*(_QWORD *)(a1 + 120), v40) )
    {
      v66 = v40 <= 0;
      v42 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v66 )
        v42 = 0x8000000000000000LL;
    }
    else
    {
      v42 = *(_QWORD *)(a1 + 120) + v40;
    }
    v43 = *(unsigned int *)(a1 + 24);
    v44 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 120) = v42;
    v45 = 0;
    if ( (_DWORD)v43 )
    {
      do
      {
        v46 = (_DWORD *)(v44 + 4LL * (unsigned int)v45);
        if ( *v46 != -1 )
          *v46 = v45;
        ++v45;
      }
      while ( v43 != v45 );
    }
    *(_BYTE *)(a1 + 200) = 0;
    v13 = *(__int64 **)(a1 + 80);
    if ( a3 )
      goto LABEL_16;
    goto LABEL_53;
  }
LABEL_15:
  *(_BYTE *)(a1 + 200) = 0;
  if ( a3 )
  {
LABEL_16:
    v67 = *v13;
    dest[0] = (void *)(a3 | 4);
    v68 = a2 | 4;
    v23 = sub_2B35AF0(a1, &v68, (__int64)dest, a4, v7, (__int64)dest);
    if ( v25 == 1 )
      *(_DWORD *)(a1 + 128) = 1;
    if ( __OFADD__(*(_QWORD *)(a1 + 120), v23) )
    {
      v66 = v23 <= 0;
      v26 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v66 )
        v26 = 0x8000000000000000LL;
    }
    else
    {
      v26 = *(_QWORD *)(a1 + 120) + v23;
    }
    *(_QWORD *)(a1 + 120) = v26;
    v27 = v67 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v67 & 4) != 0 || !v27 )
    {
      v29 = *(_DWORD *)(v27 + 120);
      if ( !v29 )
        v29 = *(_DWORD *)(v27 + 8);
    }
    else
    {
      v28 = *(_QWORD *)(v27 + 8);
      v29 = 1;
      if ( *(_BYTE *)(v28 + 8) == 17 )
        v29 = *(_DWORD *)(v28 + 32);
    }
    v30 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v7 < v29 )
      LODWORD(v7) = v29;
    if ( (_DWORD)v30 )
    {
      v31 = 0;
      do
      {
        v32 = v31;
        if ( *(_DWORD *)&a4[4 * v31] != -1 )
        {
          if ( *(_DWORD *)(a1 + 88) )
            v32 = v7 + v31;
          *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v31) = v32;
        }
        ++v31;
      }
      while ( (unsigned int)v30 != v31 );
      v30 = *(unsigned int *)(a1 + 24);
    }
    v33 = sub_2B35AF0(a1, &v67, *(_QWORD *)(a1 + 80), *(const void **)(a1 + 16), v30, v24);
    if ( v34 == 1 )
      *(_DWORD *)(a1 + 128) = 1;
    if ( __OFADD__(*(_QWORD *)(a1 + 120), v33) )
    {
      v66 = v33 <= 0;
      v35 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v66 )
        v35 = 0x8000000000000000LL;
    }
    else
    {
      v35 = *(_QWORD *)(a1 + 120) + v33;
    }
    v36 = *(unsigned int *)(a1 + 24);
    v37 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 120) = v35;
    v38 = 0;
    if ( (_DWORD)v36 )
    {
      do
      {
        v39 = (_DWORD *)(v37 + 4LL * (unsigned int)v38);
        if ( *v39 != -1 )
          *v39 = v38;
        ++v38;
      }
      while ( v36 != v38 );
    }
    return;
  }
LABEL_53:
  if ( *(_DWORD *)(a1 + 88) != 1 )
    goto LABEL_16;
  v47 = *(_DWORD *)(a2 + 120);
  if ( !v47 )
    v47 = *(_DWORD *)(a2 + 8);
  v48 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*v13 & 4) != 0 || !v48 )
  {
    v50 = *(_DWORD *)(v48 + 120);
    if ( !v50 )
      v50 = *(_DWORD *)(v48 + 8);
    if ( v47 >= v50 )
      v50 = v47;
  }
  else
  {
    v49 = *(_QWORD *)(v48 + 8);
    if ( *(_DWORD *)(v49 + 32) >= v47 )
      v47 = *(_DWORD *)(v49 + 32);
    v50 = v47;
  }
  v51 = *(unsigned int *)(a1 + 24);
  v52 = *(char **)(a1 + 16);
  if ( (_DWORD)v51 )
  {
    v53 = 0;
    do
    {
      v54 = *(_DWORD *)&a4[v53];
      if ( v54 != -1 && *(_DWORD *)&v52[v53] == -1 )
      {
        *(_DWORD *)&v52[v53] = v50 + v54;
        v52 = *(char **)(a1 + 16);
      }
      v53 += 4;
    }
    while ( v53 != 4LL * (unsigned int)v51 );
    v13 = *(__int64 **)(a1 + 80);
    v51 = *(unsigned int *)(a1 + 24);
  }
  dest[0] = (void *)(a2 | 4);
  v55 = sub_2B35AF0(a1, v13, (__int64)dest, v52, v51, (__int64)dest);
  if ( v56 == 1 )
    *(_DWORD *)(a1 + 128) = 1;
  if ( __OFADD__(*(_QWORD *)(a1 + 120), v55) )
  {
    v66 = v55 <= 0;
    v57 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v66 )
      v57 = 0x8000000000000000LL;
  }
  else
  {
    v57 = *(_QWORD *)(a1 + 120) + v55;
  }
  v58 = *(unsigned int *)(a1 + 24);
  v59 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 120) = v57;
  v60 = 0;
  if ( (_DWORD)v58 )
  {
    do
    {
      v61 = (_DWORD *)(v59 + 4LL * (unsigned int)v60);
      if ( *v61 != -1 )
        *v61 = v60;
      ++v60;
    }
    while ( v58 != v60 );
  }
}
