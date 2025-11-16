// Function: sub_1295900
// Address: 0x1295900
//
__int64 __fastcall sub_1295900(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rsi
  char *v5; // r14
  __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 i; // r13
  _BYTE *v9; // rsi
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // rsi
  unsigned int v15; // esi
  unsigned int v16; // r8d
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned int v19; // ecx
  __int64 v20; // r9
  char *v21; // r13
  _QWORD *v22; // rdx
  _BYTE *v23; // rdi
  unsigned __int64 v24; // rax
  char *v25; // rdx
  char *v26; // rsi
  size_t v27; // r15
  _BYTE *v28; // r8
  signed __int64 v29; // r10
  char *v30; // rcx
  __int64 v31; // rax
  _QWORD *v32; // r15
  __int64 v33; // rdi
  unsigned __int64 *v34; // r14
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  __int64 v37; // rsi
  _QWORD *v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rsi
  _QWORD *v45; // rax
  __int64 result; // rax
  char *v47; // rsi
  __int64 v48; // rax
  char *v49; // rcx
  __int64 v50; // rdi
  int v51; // r11d
  char *v52; // r10
  int v53; // ecx
  int v54; // edx
  _QWORD *v55; // r11
  char *v56; // r15
  char *v57; // r10
  int v58; // r9d
  int v59; // ecx
  int v60; // ecx
  int v61; // eax
  int v62; // ecx
  __int64 v63; // rsi
  unsigned int v64; // eax
  _QWORD *v65; // rdi
  int v66; // r10d
  char *v67; // r8
  int v68; // eax
  int v69; // eax
  __int64 v70; // rsi
  unsigned int v71; // edx
  _QWORD *v72; // rdi
  int v73; // r10d
  char *v74; // r8
  int v75; // edx
  int v76; // edx
  __int64 v77; // rdi
  int v78; // r10d
  unsigned int v79; // eax
  _QWORD *v80; // rsi
  int v81; // edx
  int v82; // ecx
  int v83; // r10d
  __int64 v84; // rdi
  unsigned int v85; // eax
  _QWORD *v86; // rsi
  int v87; // r10d
  __int64 v88; // r9
  char *v89; // [rsp+0h] [rbp-D0h]
  char *v90; // [rsp+0h] [rbp-D0h]
  char *v91; // [rsp+8h] [rbp-C8h]
  char *v92; // [rsp+8h] [rbp-C8h]
  unsigned int v93; // [rsp+8h] [rbp-C8h]
  unsigned int v94; // [rsp+8h] [rbp-C8h]
  unsigned int v95; // [rsp+8h] [rbp-C8h]
  __int64 v96; // [rsp+10h] [rbp-C0h]
  __int64 v97; // [rsp+10h] [rbp-C0h]
  __int64 v99; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v100; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v101; // [rsp+38h] [rbp-98h] BYREF
  __int64 v102; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v103; // [rsp+48h] [rbp-88h]
  _BYTE *v104; // [rsp+50h] [rbp-80h]
  void *src; // [rsp+60h] [rbp-70h] BYREF
  _BYTE *v106; // [rsp+68h] [rbp-68h]
  _BYTE *v107; // [rsp+70h] [rbp-60h]
  char v108[16]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v109; // [rsp+90h] [rbp-40h]

  v4 = a2[6];
  v102 = 0;
  v103 = 0;
  v104 = 0;
  src = 0;
  v106 = 0;
  v107 = 0;
  v5 = sub_128F980(a1, v4);
  v6 = a2[10];
  v7 = 0;
  for ( i = *(_QWORD *)(v6 + 16); i; v106 = v11 + 8 )
  {
    while ( 1 )
    {
      v12 = sub_127F610(*(_QWORD *)(a1 + 32), *(const __m128i **)(i + 8), 0);
      v9 = v103;
      if ( *(_BYTE *)(v12 + 16) != 13 )
        v12 = 0;
      v100 = v12;
      if ( v103 == v104 )
      {
        sub_1291F00((__int64)&v102, v103, &v100);
      }
      else
      {
        if ( v103 )
        {
          *(_QWORD *)v103 = v12;
          v9 = v103;
        }
        v103 = v9 + 8;
      }
      v10 = sub_12A4D50(a1, "switch_case.target", 0, 0);
      v11 = v106;
      v99 = v10;
      if ( v106 != v107 )
        break;
      ++v7;
      sub_1292090((__int64)&src, v106, &v99);
      i = *(_QWORD *)(i + 32);
      if ( !i )
        goto LABEL_15;
    }
    if ( v106 )
    {
      *(_QWORD *)v106 = v10;
      v11 = v106;
    }
    i = *(_QWORD *)(i + 32);
    ++v7;
  }
LABEL_15:
  v13 = sub_12A4D50(a1, "switch_case.default_target", 0, 0);
  v14 = v106;
  v99 = v13;
  if ( v106 == v107 )
  {
    sub_1292090((__int64)&src, v106, &v99);
  }
  else
  {
    if ( v106 )
    {
      *(_QWORD *)v106 = v13;
      v14 = v106;
    }
    v106 = v14 + 8;
  }
  v15 = *(_DWORD *)(a1 + 432);
  v96 = a1 + 408;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 408);
    goto LABEL_92;
  }
  v16 = v15 - 1;
  v17 = *(_QWORD *)(a1 + 416);
  v18 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
  v19 = (v15 - 1) & v18;
  v20 = 32LL * v19;
  v21 = (char *)(v17 + v20);
  v22 = *(_QWORD **)(v17 + v20);
  if ( v22 == a2 )
  {
    if ( &src != (void **)(v21 + 8) )
      goto LABEL_22;
    goto LABEL_31;
  }
  v93 = (v15 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v55 = *(_QWORD **)(v17 + 32LL * v93);
  v56 = (char *)(v17 + v20);
  v57 = 0;
  v58 = 1;
  while ( 1 )
  {
    if ( v55 == (_QWORD *)-8LL )
    {
      v59 = *(_DWORD *)(a1 + 424);
      if ( !v57 )
        v57 = v56;
      ++*(_QWORD *)(a1 + 408);
      v60 = v59 + 1;
      v21 = v57;
      if ( 4 * v60 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 428) - v60 <= v15 >> 3 )
        {
          v94 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
          sub_12953C0(v96, v15);
          v75 = *(_DWORD *)(a1 + 432);
          if ( !v75 )
            goto LABEL_138;
          v76 = v75 - 1;
          v77 = *(_QWORD *)(a1 + 416);
          v78 = 1;
          v74 = 0;
          v79 = v76 & v94;
          v60 = *(_DWORD *)(a1 + 424) + 1;
          v21 = (char *)(v77 + 32LL * (v76 & v94));
          v80 = *(_QWORD **)v21;
          if ( *(_QWORD **)v21 != a2 )
          {
            while ( v80 != (_QWORD *)-8LL )
            {
              if ( !v74 && v80 == (_QWORD *)-16LL )
                v74 = v21;
              v79 = v76 & (v78 + v79);
              v21 = (char *)(v77 + 32LL * v79);
              v80 = *(_QWORD **)v21;
              if ( *(_QWORD **)v21 == a2 )
                goto LABEL_79;
              ++v78;
            }
LABEL_96:
            if ( v74 )
              v21 = v74;
            goto LABEL_79;
          }
        }
        goto LABEL_79;
      }
LABEL_92:
      sub_12953C0(v96, 2 * v15);
      v68 = *(_DWORD *)(a1 + 432);
      if ( !v68 )
        goto LABEL_138;
      v69 = v68 - 1;
      v70 = *(_QWORD *)(a1 + 416);
      v71 = v69 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v60 = *(_DWORD *)(a1 + 424) + 1;
      v21 = (char *)(v70 + 32LL * v71);
      v72 = *(_QWORD **)v21;
      if ( *(_QWORD **)v21 != a2 )
      {
        v73 = 1;
        v74 = 0;
        while ( v72 != (_QWORD *)-8LL )
        {
          if ( v72 == (_QWORD *)-16LL && !v74 )
            v74 = v21;
          v71 = v69 & (v73 + v71);
          v21 = (char *)(v70 + 32LL * v71);
          v72 = *(_QWORD **)v21;
          if ( *(_QWORD **)v21 == a2 )
            goto LABEL_79;
          ++v73;
        }
        goto LABEL_96;
      }
LABEL_79:
      *(_DWORD *)(a1 + 424) = v60;
      if ( *(_QWORD *)v21 != -8 )
        --*(_DWORD *)(a1 + 428);
      *((_QWORD *)v21 + 1) = 0;
      *((_QWORD *)v21 + 2) = 0;
      *(_QWORD *)v21 = a2;
      *((_QWORD *)v21 + 3) = 0;
      if ( &src != (void **)(v21 + 8) )
      {
        v24 = 0;
        v23 = 0;
        goto LABEL_23;
      }
      goto LABEL_29;
    }
    if ( v57 || v55 != (_QWORD *)-16LL )
      v56 = v57;
    v87 = v58 + 1;
    v88 = v16 & (v93 + v58);
    v93 = v88;
    v90 = (char *)(v17 + 32 * v88);
    v55 = *(_QWORD **)v90;
    if ( *(_QWORD **)v90 == a2 )
      break;
    v58 = v87;
    v57 = v56;
    v56 = v90;
  }
  if ( &src == (void **)(v90 + 8) )
    goto LABEL_63;
  v21 = (char *)(v17 + 32 * v88);
LABEL_22:
  v23 = (_BYTE *)*((_QWORD *)v21 + 1);
  v24 = *((_QWORD *)v21 + 3) - (_QWORD)v23;
LABEL_23:
  v25 = v106;
  v26 = (char *)src;
  v27 = v106 - (_BYTE *)src;
  if ( v24 < v106 - (_BYTE *)src )
  {
    if ( v27 )
    {
      if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v23, src, v106);
      v89 = (char *)src;
      v91 = v106;
      v48 = sub_22077B0(v106 - (_BYTE *)src);
      v25 = v91;
      v26 = v89;
      v49 = (char *)v48;
    }
    else
    {
      v49 = 0;
    }
    if ( v25 != v26 )
      v49 = (char *)memcpy(v49, v26, v27);
    v50 = *((_QWORD *)v21 + 1);
    if ( v50 )
    {
      v92 = v49;
      j_j___libc_free_0(v50, *((_QWORD *)v21 + 3) - v50);
      v49 = v92;
    }
    *((_QWORD *)v21 + 1) = v49;
    v30 = &v49[v27];
    *((_QWORD *)v21 + 3) = v30;
  }
  else
  {
    v28 = (_BYTE *)*((_QWORD *)v21 + 2);
    v29 = v28 - v23;
    if ( v27 <= v28 - v23 )
    {
      if ( v106 != src )
      {
        memmove(v23, src, v106 - (_BYTE *)src);
        v23 = (_BYTE *)*((_QWORD *)v21 + 1);
      }
      goto LABEL_27;
    }
    if ( v29 )
    {
      memmove(v23, src, *((_QWORD *)v21 + 2) - (_QWORD)v23);
      v28 = (_BYTE *)*((_QWORD *)v21 + 2);
      v23 = (_BYTE *)*((_QWORD *)v21 + 1);
      v25 = v106;
      v26 = (char *)src;
      v29 = v28 - v23;
    }
    v47 = &v26[v29];
    if ( v47 == v25 )
    {
LABEL_27:
      v30 = &v23[v27];
    }
    else
    {
      memmove(v28, v47, v25 - v47);
      v30 = (char *)(v27 + *((_QWORD *)v21 + 1));
    }
  }
  *((_QWORD *)v21 + 2) = v30;
LABEL_29:
  v15 = *(_DWORD *)(a1 + 432);
  if ( v15 )
  {
    v16 = v15 - 1;
    v17 = *(_QWORD *)(a1 + 416);
    v18 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
    v19 = (v15 - 1) & v18;
    v21 = (char *)(v17 + 32LL * v19);
    v22 = *(_QWORD **)v21;
    if ( *(_QWORD **)v21 == a2 )
      goto LABEL_31;
LABEL_63:
    v51 = 1;
    v52 = 0;
    while ( v22 != (_QWORD *)-8LL )
    {
      if ( !v52 && v22 == (_QWORD *)-16LL )
        v52 = v21;
      v19 = v16 & (v51 + v19);
      v21 = (char *)(v17 + 32LL * v19);
      v22 = *(_QWORD **)v21;
      if ( *(_QWORD **)v21 == a2 )
        goto LABEL_31;
      ++v51;
    }
    v53 = *(_DWORD *)(a1 + 424);
    if ( v52 )
      v21 = v52;
    ++*(_QWORD *)(a1 + 408);
    v54 = v53 + 1;
    if ( 4 * (v53 + 1) < 3 * v15 )
    {
      if ( v15 - (v54 + *(_DWORD *)(a1 + 428)) > v15 >> 3 )
        goto LABEL_69;
      v95 = v18;
      sub_12953C0(v96, v15);
      v81 = *(_DWORD *)(a1 + 432);
      if ( v81 )
      {
        v82 = v81 - 1;
        v83 = 1;
        v67 = 0;
        v84 = *(_QWORD *)(a1 + 416);
        v85 = v82 & v95;
        v54 = *(_DWORD *)(a1 + 424) + 1;
        v21 = (char *)(v84 + 32LL * (v82 & v95));
        v86 = *(_QWORD **)v21;
        if ( *(_QWORD **)v21 != a2 )
        {
          while ( v86 != (_QWORD *)-8LL )
          {
            if ( !v67 && v86 == (_QWORD *)-16LL )
              v67 = v21;
            v85 = v82 & (v83 + v85);
            v21 = (char *)(v84 + 32LL * v85);
            v86 = *(_QWORD **)v21;
            if ( *(_QWORD **)v21 == a2 )
              goto LABEL_69;
            ++v83;
          }
          goto LABEL_88;
        }
        goto LABEL_69;
      }
LABEL_138:
      ++*(_DWORD *)(a1 + 424);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 408);
  }
  sub_12953C0(v96, 2 * v15);
  v61 = *(_DWORD *)(a1 + 432);
  if ( !v61 )
    goto LABEL_138;
  v62 = v61 - 1;
  v63 = *(_QWORD *)(a1 + 416);
  v64 = (v61 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v54 = *(_DWORD *)(a1 + 424) + 1;
  v21 = (char *)(v63 + 32LL * v64);
  v65 = *(_QWORD **)v21;
  if ( *(_QWORD **)v21 != a2 )
  {
    v66 = 1;
    v67 = 0;
    while ( v65 != (_QWORD *)-8LL )
    {
      if ( v65 == (_QWORD *)-16LL && !v67 )
        v67 = v21;
      v64 = v62 & (v66 + v64);
      v21 = (char *)(v63 + 32LL * v64);
      v65 = *(_QWORD **)v21;
      if ( *(_QWORD **)v21 == a2 )
        goto LABEL_69;
      ++v66;
    }
LABEL_88:
    if ( v67 )
      v21 = v67;
  }
LABEL_69:
  *(_DWORD *)(a1 + 424) = v54;
  if ( *(_QWORD *)v21 != -8 )
    --*(_DWORD *)(a1 + 428);
  *((_QWORD *)v21 + 1) = 0;
  *((_QWORD *)v21 + 2) = 0;
  *(_QWORD *)v21 = a2;
  *((_QWORD *)v21 + 3) = 0;
LABEL_31:
  v109 = 257;
  v97 = v99;
  v31 = sub_1648B60(64);
  v32 = (_QWORD *)v31;
  if ( v31 )
    sub_15FFAB0(v31, v5, v97, v7, 0);
  v33 = *(_QWORD *)(a1 + 56);
  if ( v33 )
  {
    v34 = *(unsigned __int64 **)(a1 + 64);
    sub_157E9D0(v33 + 40, v32);
    v35 = v32[3];
    v36 = *v34;
    v32[4] = v34;
    v36 &= 0xFFFFFFFFFFFFFFF8LL;
    v32[3] = v36 | v35 & 7;
    *(_QWORD *)(v36 + 8) = v32 + 3;
    *v34 = *v34 & 7 | (unsigned __int64)(v32 + 3);
  }
  sub_164B780(v32, v108);
  v37 = *(_QWORD *)(a1 + 48);
  if ( v37 )
  {
    v101 = *(_QWORD *)(a1 + 48);
    sub_1623A60(&v101, v37, 2);
    v38 = v32 + 6;
    if ( v32[6] )
    {
      sub_161E7C0(v32 + 6);
      v38 = v32 + 6;
    }
    v39 = v101;
    v32[6] = v101;
    if ( v39 )
      sub_1623210(&v101, v39, v38);
  }
  if ( v7 )
  {
    v40 = v7 - 1;
    v41 = 0;
    v42 = 8 * v40 + 8;
    do
    {
      v43 = *(_QWORD *)(*((_QWORD *)v21 + 1) + v41);
      v44 = *(_QWORD *)(v102 + v41);
      v41 += 8;
      sub_15FFFB0(v32, v44, v43);
    }
    while ( v42 != v41 );
  }
  v45 = (_QWORD *)sub_12A4D50(a1, "switch_child_entry", 0, 0);
  sub_1290AF0((_QWORD *)a1, v45, 0);
  sub_1296350(a1, a2[9]);
  result = a2[10];
  if ( !*(_QWORD *)(result + 8) )
    result = sub_1290AF0((_QWORD *)a1, *((_QWORD **)v106 - 1), 0);
  if ( src )
    result = j_j___libc_free_0(src, v107 - (_BYTE *)src);
  if ( v102 )
    return j_j___libc_free_0(v102, &v104[-v102]);
  return result;
}
