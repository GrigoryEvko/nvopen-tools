// Function: sub_372D1A0
// Address: 0x372d1a0
//
void __fastcall sub_372D1A0(__int128 a1, int a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 *v12; // r12
  __int64 v13; // r14
  _QWORD *v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r15
  int v31; // eax
  int *v32; // r8
  char v33; // r14
  int *v34; // rbx
  unsigned int v35; // r15d
  _DWORD *v36; // rax
  _DWORD *v37; // rdx
  unsigned __int64 v38; // rax
  int *v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // r15
  unsigned __int64 *v45; // rbx
  int *v46; // r12
  unsigned __int64 v47; // rax
  char *v48; // rdi
  __int64 v49; // rcx
  char *v50; // rdx
  int *v51; // r14
  int *v52; // r13
  int *v53; // rdx
  bool v54; // al
  int *v55; // rsi
  __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // rcx
  int *v59; // rdi
  int *v60; // rax
  _QWORD *v62; // [rsp+38h] [rbp-178h]
  __int64 v63; // [rsp+40h] [rbp-170h]
  __int64 *v64; // [rsp+48h] [rbp-168h]
  int *v66; // [rsp+60h] [rbp-150h]
  char v67; // [rsp+68h] [rbp-148h]
  unsigned __int64 *v68; // [rsp+68h] [rbp-148h]
  __m128i v69; // [rsp+70h] [rbp-140h] BYREF
  unsigned int v70; // [rsp+8Ch] [rbp-124h] BYREF
  __m128i v71[2]; // [rsp+90h] [rbp-120h] BYREF
  unsigned __int64 *v72; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-F8h]
  _BYTE v74[32]; // [rsp+C0h] [rbp-F0h] BYREF
  int *v75; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+E8h] [rbp-C8h]
  _BYTE v77[24]; // [rsp+F0h] [rbp-C0h] BYREF
  int v78; // [rsp+108h] [rbp-A8h] BYREF
  unsigned __int64 v79; // [rsp+110h] [rbp-A0h]
  int *v80; // [rsp+118h] [rbp-98h]
  int *v81; // [rsp+120h] [rbp-90h]
  __int64 v82; // [rsp+128h] [rbp-88h]
  _BYTE *v83; // [rsp+130h] [rbp-80h] BYREF
  __int64 v84; // [rsp+138h] [rbp-78h]
  _BYTE v85[24]; // [rsp+140h] [rbp-70h] BYREF
  int v86; // [rsp+158h] [rbp-58h] BYREF
  unsigned __int64 v87; // [rsp+160h] [rbp-50h]
  int *v88; // [rsp+168h] [rbp-48h]
  int *v89; // [rsp+170h] [rbp-40h]
  __int64 v90; // [rsp+178h] [rbp-38h]

  v69 = (__m128i)a1;
  v8 = sub_372D0C0(a5, a1, *((__int64 *)&a1 + 1), a3);
  v10 = (__int64)(a4 + 1);
  v78 = 0;
  v63 = v8;
  v72 = (unsigned __int64 *)v74;
  v73 = 0x400000000LL;
  v76 = 0x400000000LL;
  v84 = 0x400000000LL;
  v75 = (int *)v77;
  v88 = &v86;
  v89 = &v86;
  v11 = (_QWORD *)a4[2];
  v80 = &v78;
  v81 = &v78;
  v79 = 0;
  v82 = 0;
  v83 = v85;
  v86 = 0;
  v87 = 0;
  v90 = 0;
  v62 = a4 + 1;
  if ( v11 )
  {
    do
    {
      while ( v11[4] >= v69.m128i_i64[0] && (v11[4] != v69.m128i_i64[0] || v11[5] >= v69.m128i_i64[1]) )
      {
        v10 = (__int64)v11;
        v11 = (_QWORD *)v11[2];
        if ( !v11 )
          goto LABEL_8;
      }
      v11 = (_QWORD *)v11[3];
    }
    while ( v11 );
LABEL_8:
    if ( v62 != (_QWORD *)v10
      && *(_QWORD *)(v10 + 32) <= v69.m128i_i64[0]
      && (*(_QWORD *)(v10 + 32) != v69.m128i_i64[0] || *(_QWORD *)(v10 + 40) <= v69.m128i_i64[1]) )
    {
      goto LABEL_11;
    }
  }
  else
  {
    v10 = (__int64)(a4 + 1);
  }
  v71[0].m128i_i64[0] = (__int64)&v69;
  v10 = sub_372B810(a4, (_QWORD *)v10, (const __m128i **)v71);
LABEL_11:
  if ( *(_QWORD *)(v10 + 112) )
  {
    v67 = 0;
    v12 = *(__int64 **)(v10 + 96);
    v64 = (__int64 *)(v10 + 80);
  }
  else
  {
    v12 = *(__int64 **)(v10 + 48);
    v67 = 1;
    v64 = &v12[*(unsigned int *)(v10 + 56)];
  }
  if ( v67 )
    goto LABEL_34;
  while ( v64 != v12 )
  {
    v13 = v12[4];
    v71[0] = _mm_load_si128(&v69);
    v14 = (_QWORD *)(*(_QWORD *)sub_372CB80(a5, v71) + 16 * v13);
    if ( sub_2E89C10(*v14 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_37;
LABEL_16:
    v17 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
    v18 = *(_QWORD *)(v17 + 32);
    if ( *(_WORD *)(v17 + 68) == 14 )
    {
      v9 = v18 + 40;
      v20 = *(_QWORD *)(v17 + 32);
    }
    else
    {
      v19 = 5LL * (*(_DWORD *)(v17 + 40) & 0xFFFFFF);
      v20 = v18 + 80;
      v19 *= 8;
      v9 = v18 + v19;
      v18 += 80;
      v15 = 0xCCCCCCCCCCCCCCCDLL * ((v19 - 80) >> 3);
      v21 = v15 >> 2;
      if ( v15 >> 2 > 0 )
      {
        while ( *(_BYTE *)v18 || a2 != *(_DWORD *)(v18 + 8) )
        {
          if ( !*(_BYTE *)(v18 + 40) && a2 == *(_DWORD *)(v18 + 48) )
          {
            v18 += 40;
            goto LABEL_39;
          }
          if ( !*(_BYTE *)(v18 + 80) && a2 == *(_DWORD *)(v18 + 88) )
          {
            v18 += 80;
            goto LABEL_39;
          }
          if ( !*(_BYTE *)(v18 + 120) && a2 == *(_DWORD *)(v18 + 128) )
          {
            v18 += 120;
            goto LABEL_39;
          }
          v18 += 160;
          if ( !--v21 )
          {
            v15 = 0xCCCCCCCCCCCCCCCDLL * ((v9 - v18) >> 3);
            goto LABEL_28;
          }
        }
        goto LABEL_39;
      }
LABEL_28:
      if ( v15 == 2 )
        goto LABEL_153;
      if ( v15 == 3 )
      {
        if ( !*(_BYTE *)v18 && a2 == *(_DWORD *)(v18 + 8) )
          goto LABEL_39;
        v18 += 40;
LABEL_153:
        if ( !*(_BYTE *)v18 && a2 == *(_DWORD *)(v18 + 8) )
          goto LABEL_39;
        v18 += 40;
        goto LABEL_47;
      }
      if ( v15 != 1 )
      {
        if ( v9 == v20 )
          goto LABEL_32;
LABEL_41:
        v22 = v20;
        v23 = v9;
        do
        {
          if ( !*(_BYTE *)v22 && *(_DWORD *)(v22 + 8) )
          {
            v70 = *(_DWORD *)(v22 + 8);
            sub_372C4F0((__int64)v71, (__int64)&v83, &v70, v15, v16);
          }
          v22 += 40;
        }
        while ( v23 != v22 );
        goto LABEL_32;
      }
    }
LABEL_47:
    if ( *(_BYTE *)v18 || a2 != *(_DWORD *)(v18 + 8) )
    {
LABEL_40:
      if ( v9 == v20 )
        goto LABEL_32;
      goto LABEL_41;
    }
LABEL_39:
    if ( v18 == v9 )
      goto LABEL_40;
    v24 = (unsigned int)v73;
    v25 = (unsigned int)v73 + 1LL;
    if ( v25 > HIDWORD(v73) )
    {
      sub_C8D5F0((__int64)&v72, v74, v25, 8u, v16, v9);
      v24 = (unsigned int)v73;
    }
    v72[v24] = v13;
    LODWORD(v73) = v73 + 1;
    sub_372AB10((__int64)v14, v63);
    v28 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
    v29 = *(_QWORD *)(v28 + 32);
    if ( *(_WORD *)(v28 + 68) == 14 )
    {
      v30 = v29 + 40;
    }
    else
    {
      v30 = v29 + 40LL * (*(_DWORD *)(v28 + 40) & 0xFFFFFF);
      v29 += 80;
    }
    for ( ; v30 != v29; v29 += 40 )
    {
      if ( !*(_BYTE *)v29 )
      {
        v31 = *(_DWORD *)(v29 + 8);
        if ( a2 != v31 )
        {
          if ( v31 )
          {
            v70 = *(_DWORD *)(v29 + 8);
            sub_372C4F0((__int64)v71, (__int64)&v75, &v70, v26, v27);
          }
        }
      }
    }
LABEL_32:
    if ( v67 )
    {
      while ( 1 )
      {
        ++v12;
LABEL_34:
        if ( v64 == v12 )
          goto LABEL_64;
        v13 = *v12;
        v71[0] = _mm_load_si128(&v69);
        v14 = (_QWORD *)(*(_QWORD *)sub_372CB80(a5, v71) + 16 * v13);
        if ( !sub_2E89C10(*v14 & 0xFFFFFFFFFFFFFFF8LL) )
          goto LABEL_16;
      }
    }
LABEL_37:
    v12 = (__int64 *)sub_220EF30((__int64)v12);
  }
LABEL_64:
  if ( v82 )
  {
    v32 = v80;
    v34 = &v78;
    v33 = 0;
  }
  else
  {
    v32 = v75;
    v33 = 1;
    v34 = &v75[(unsigned int)v76];
  }
  if ( v33 )
    goto LABEL_76;
  while ( 2 )
  {
    if ( v34 == v32 )
      goto LABEL_89;
    v35 = v32[8];
    if ( !v90 )
      goto LABEL_69;
LABEL_78:
    v38 = v87;
    if ( !v87 )
      goto LABEL_85;
    v39 = &v86;
    do
    {
      while ( 1 )
      {
        v40 = *(_QWORD *)(v38 + 16);
        v41 = *(_QWORD *)(v38 + 24);
        if ( v35 <= *(_DWORD *)(v38 + 32) )
          break;
        v38 = *(_QWORD *)(v38 + 24);
        if ( !v41 )
          goto LABEL_83;
      }
      v39 = (int *)v38;
      v38 = *(_QWORD *)(v38 + 16);
    }
    while ( v40 );
LABEL_83:
    if ( v39 == &v86 || v35 < v39[8] )
    {
LABEL_85:
      v42 = *(unsigned int *)(a6 + 8);
      if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        v66 = v32;
        sub_C8D5F0(a6, (const void *)(a6 + 16), v42 + 1, 4u, (__int64)v32, v9);
        v42 = *(unsigned int *)(a6 + 8);
        v32 = v66;
      }
      *(_DWORD *)(*(_QWORD *)a6 + 4 * v42) = v35;
      ++*(_DWORD *)(a6 + 8);
      if ( v33 )
        goto LABEL_75;
LABEL_88:
      v32 = (int *)sub_220EF30((__int64)v32);
      continue;
    }
    break;
  }
  while ( 1 )
  {
    if ( !v33 )
      goto LABEL_88;
LABEL_75:
    ++v32;
LABEL_76:
    if ( v34 == v32 )
      break;
    v35 = *v32;
    if ( v90 )
      goto LABEL_78;
LABEL_69:
    v36 = v83;
    v37 = &v83[4 * (unsigned int)v84];
    if ( v83 != (_BYTE *)v37 )
    {
      while ( v35 != *v36 )
      {
        if ( v37 == ++v36 )
          goto LABEL_85;
      }
      if ( v37 != v36 )
        continue;
    }
    goto LABEL_85;
  }
LABEL_89:
  v43 = (_QWORD *)a4[2];
  if ( !v43 )
  {
    v44 = (__int64)v62;
    goto LABEL_98;
  }
  v44 = (__int64)v62;
  do
  {
    while ( v43[4] >= v69.m128i_i64[0] && (v43[4] != v69.m128i_i64[0] || v43[5] >= v69.m128i_i64[1]) )
    {
      v44 = (__int64)v43;
      v43 = (_QWORD *)v43[2];
      if ( !v43 )
        goto LABEL_96;
    }
    v43 = (_QWORD *)v43[3];
  }
  while ( v43 );
LABEL_96:
  if ( v62 == (_QWORD *)v44
    || *(_QWORD *)(v44 + 32) > v69.m128i_i64[0]
    || *(_QWORD *)(v44 + 32) == v69.m128i_i64[0] && *(_QWORD *)(v44 + 40) > v69.m128i_i64[1] )
  {
LABEL_98:
    v71[0].m128i_i64[0] = (__int64)&v69;
    v44 = sub_372B810(a4, (_QWORD *)v44, (const __m128i **)v71);
  }
  v45 = v72;
  v46 = (int *)(v44 + 80);
  v68 = &v72[(unsigned int)v73];
  if ( v72 != v68 )
  {
    while ( 2 )
    {
      v47 = *v45;
      if ( !*(_QWORD *)(v44 + 112) )
      {
        v48 = *(char **)(v44 + 48);
        v49 = *(unsigned int *)(v44 + 56);
        v50 = &v48[8 * v49];
        if ( v48 != v50 )
        {
          while ( v47 != *(_QWORD *)v48 )
          {
            v48 += 8;
            if ( v50 == v48 )
              goto LABEL_109;
          }
          if ( v48 != v50 )
          {
            if ( v50 != v48 + 8 )
            {
              memmove(v48, v48 + 8, v50 - (v48 + 8));
              LODWORD(v49) = *(_DWORD *)(v44 + 56);
            }
            *(_DWORD *)(v44 + 56) = v49 - 1;
          }
        }
        goto LABEL_109;
      }
      v51 = (int *)(v44 + 80);
      if ( *(_QWORD *)(v44 + 88) )
      {
        v52 = *(int **)(v44 + 88);
        while ( 1 )
        {
          while ( v47 > *((_QWORD *)v52 + 4) )
          {
            v52 = (int *)*((_QWORD *)v52 + 3);
            if ( !v52 )
              goto LABEL_123;
          }
          v53 = (int *)*((_QWORD *)v52 + 2);
          if ( v47 >= *((_QWORD *)v52 + 4) )
            break;
          v51 = v52;
          v52 = (int *)*((_QWORD *)v52 + 2);
          if ( !v53 )
          {
LABEL_123:
            v54 = v46 == v51;
            goto LABEL_124;
          }
        }
        v55 = (int *)*((_QWORD *)v52 + 3);
        if ( v55 )
        {
          do
          {
            while ( 1 )
            {
              v56 = *((_QWORD *)v55 + 2);
              v57 = *((_QWORD *)v55 + 3);
              if ( v47 < *((_QWORD *)v55 + 4) )
                break;
              v55 = (int *)*((_QWORD *)v55 + 3);
              if ( !v57 )
                goto LABEL_132;
            }
            v51 = v55;
            v55 = (int *)*((_QWORD *)v55 + 2);
          }
          while ( v56 );
        }
LABEL_132:
        while ( v53 )
        {
          while ( 1 )
          {
            v58 = *((_QWORD *)v53 + 3);
            if ( v47 <= *((_QWORD *)v53 + 4) )
              break;
            v53 = (int *)*((_QWORD *)v53 + 3);
            if ( !v58 )
              goto LABEL_135;
          }
          v52 = v53;
          v53 = (int *)*((_QWORD *)v53 + 2);
        }
LABEL_135:
        if ( *(int **)(v44 + 96) != v52 || v51 != v46 )
        {
          for ( ; v51 != v52; --*(_QWORD *)(v44 + 112) )
          {
            v59 = v52;
            v52 = (int *)sub_220EF30((__int64)v52);
            v60 = sub_220F330(v59, v46);
            j_j___libc_free_0((unsigned __int64)v60);
          }
          goto LABEL_109;
        }
      }
      else
      {
        v54 = 1;
LABEL_124:
        if ( *(int **)(v44 + 96) != v51 || !v54 )
        {
LABEL_109:
          if ( v68 == ++v45 )
            goto LABEL_110;
          continue;
        }
      }
      break;
    }
    sub_372A530(*(_QWORD *)(v44 + 88));
    *(_QWORD *)(v44 + 88) = 0;
    *(_QWORD *)(v44 + 96) = v46;
    *(_QWORD *)(v44 + 104) = v46;
    *(_QWORD *)(v44 + 112) = 0;
    goto LABEL_109;
  }
LABEL_110:
  sub_372A360(v87);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  sub_372A360(v79);
  if ( v75 != (int *)v77 )
    _libc_free((unsigned __int64)v75);
  if ( v72 != (unsigned __int64 *)v74 )
    _libc_free((unsigned __int64)v72);
}
