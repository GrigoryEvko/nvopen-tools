// Function: sub_144F630
// Address: 0x144f630
//
__int64 __fastcall sub_144F630(__int64 *a1, char *a2)
{
  char *v2; // r15
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  void *v6; // rdx
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // r12
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // r10
  __int64 v14; // rdx
  unsigned int v15; // r15d
  unsigned int v16; // r14d
  __int64 v17; // r12
  unsigned int v18; // ebx
  unsigned int v19; // eax
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // rdx
  const char *v23; // rdx
  _BYTE *v24; // r14
  size_t v25; // r13
  const char *v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rdi
  unsigned int v29; // ebx
  __int64 v30; // r13
  __int64 i; // rdi
  __int64 v32; // r14
  int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // r14
  __m128i *v36; // rdx
  const char **v37; // rdi
  __int64 v38; // r12
  _BYTE *v39; // rax
  __int64 v40; // r12
  __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // r13
  unsigned int v46; // ebx
  __int64 v47; // rdi
  __int64 v48; // r12
  __int64 v49; // r13
  __int64 v50; // rdi
  int v51; // eax
  __int64 v52; // rdi
  unsigned int v53; // r14d
  unsigned __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // r13
  __int64 v59; // rdi
  int v60; // eax
  unsigned __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // r13
  __int64 v66; // rdi
  int v67; // eax
  __int64 result; // rax
  __int64 v69; // rdx
  __m128i *v70; // rdx
  __m128i si128; // xmm0
  __int64 v72; // rdi
  _BYTE *v73; // rax
  __int64 v74; // r12
  _BYTE *v75; // rax
  __int64 v76; // rdi
  _BYTE *v77; // rax
  _QWORD *v78; // rdi
  unsigned int v79; // [rsp-10h] [rbp-140h]
  __int64 v81; // [rsp+20h] [rbp-110h]
  char v82; // [rsp+20h] [rbp-110h]
  int v83; // [rsp+28h] [rbp-108h]
  int v85; // [rsp+48h] [rbp-E8h]
  _QWORD v86[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v87[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v88; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v89; // [rsp+98h] [rbp-98h]
  _QWORD v90[2]; // [rsp+A0h] [rbp-90h] BYREF
  char *v91; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v92; // [rsp+B8h] [rbp-78h]
  _QWORD v93[2]; // [rsp+C0h] [rbp-70h] BYREF
  const char *v94; // [rsp+D0h] [rbp-60h] BYREF
  __m128i *v95; // [rsp+D8h] [rbp-58h]
  __m128i *v96; // [rsp+E0h] [rbp-50h] BYREF
  __m128i *v97; // [rsp+E8h] [rbp-48h]
  int v98; // [rsp+F0h] [rbp-40h]
  const char **v99; // [rsp+F8h] [rbp-38h]

  v2 = a2;
  v3 = *a1;
  v4 = *(_QWORD *)(v3 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 16) - v4) <= 4 )
  {
    v3 = sub_16E7EE0(v3, "\tNode", 5);
  }
  else
  {
    *(_DWORD *)v4 = 1685016073;
    *(_BYTE *)(v4 + 4) = 101;
    *(_QWORD *)(v3 + 24) += 5LL;
  }
  v5 = sub_16E7B40(v3, a2);
  v6 = *(void **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0xEu )
  {
    sub_16E7EE0(v5, " [shape=record,", 15);
  }
  else
  {
    qmemcpy(v6, " [shape=record,", 15);
    *(_QWORD *)(v5 + 24) += 15LL;
  }
  v7 = *a1;
  v8 = *(_QWORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v8 <= 7u )
  {
    sub_16E7EE0(v7, "label=\"{", 8);
  }
  else
  {
    *v8 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v7 + 24) += 8LL;
  }
  v9 = *a1;
  if ( (*(_QWORD *)a2 & 4) != 0 )
  {
    v88 = v90;
    strcpy((char *)v90, "Not implemented");
    v89 = 15;
    goto LABEL_36;
  }
  v10 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *((_BYTE *)a1 + 16) )
  {
    sub_1649960(*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v22 )
    {
      LOBYTE(v93[0]) = 0;
      v91 = (char *)v93;
      v99 = (const char **)&v91;
      v94 = (const char *)&unk_49EFBE0;
      v92 = 0;
      v98 = 1;
      v97 = 0;
      v96 = 0;
      v95 = 0;
      sub_15537D0(v10, &v94, 0);
      if ( v97 != v95 )
        sub_16E7BA0(&v94);
      v88 = v90;
      sub_144C790((__int64 *)&v88, *v99, (__int64)&v99[1][(_QWORD)*v99]);
      goto LABEL_34;
    }
    v24 = (_BYTE *)sub_1649960(v10);
    v25 = (size_t)v23;
    if ( !v24 )
    {
      LOBYTE(v90[0]) = 0;
      v88 = v90;
      v89 = 0;
      goto LABEL_36;
    }
    v94 = v23;
    v26 = v23;
    v88 = v90;
    if ( (unsigned __int64)v23 > 0xF )
    {
      v88 = (_QWORD *)sub_22409D0(&v88, &v94, 0);
      v78 = v88;
      v90[0] = v94;
    }
    else
    {
      if ( v23 == (const char *)1 )
      {
        LOBYTE(v90[0]) = *v24;
        v27 = v90;
LABEL_137:
        v89 = (__int64)v26;
        v26[(_QWORD)v27] = 0;
        goto LABEL_36;
      }
      if ( !v23 )
      {
        v27 = v90;
        goto LABEL_137;
      }
      v78 = v90;
    }
    memcpy(v78, v24, v25);
    v26 = v94;
    v27 = v88;
    goto LABEL_137;
  }
  v11 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  LOBYTE(v93[0]) = 0;
  v91 = (char *)v93;
  v92 = 0;
  v98 = 1;
  v97 = 0;
  v96 = 0;
  v95 = 0;
  v94 = (const char *)&unk_49EFBE0;
  v99 = (const char **)&v91;
  sub_1649960(v11);
  if ( !v12 )
  {
    sub_15537D0(v10, &v94, 0);
    if ( v96 == v97 )
    {
      sub_16E7EE0(&v94, ":", 1);
    }
    else
    {
      v97->m128i_i8[0] = 58;
      v97 = (__m128i *)((char *)v97 + 1);
    }
  }
  sub_155C2B0(v10, &v94, 0);
  if ( v97 != v95 )
    sub_16E7BA0(&v94);
  v88 = v90;
  sub_144C790((__int64 *)&v88, *v99, (__int64)&v99[1][(_QWORD)*v99]);
  if ( *(_BYTE *)v88 == 10 )
    sub_2240CE0(&v88, 0, 1);
  v13 = v89;
  if ( v89 )
  {
    v81 = v9;
    v14 = (__int64)v88;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    do
    {
      v20 = *(_BYTE *)(v14 + v17);
      if ( v20 == 10 )
      {
        *(_BYTE *)(v14 + v17) = 92;
        v15 = 0;
        v18 = 0;
        sub_2240FD0(&v88, v17 + 1, 0, 1, 108);
        v19 = v16++;
        v14 = (__int64)v88;
        v13 = v89;
      }
      else if ( v20 == 59 )
      {
        v69 = (unsigned int)sub_22417D0(&v88, 10, v16 + 1);
        if ( v69 == v89 )
        {
          v89 = v17;
          *((_BYTE *)v88 + v17) = 0;
        }
        else
        {
          sub_2240CE0(&v88, v17, v69 - v17);
        }
        v14 = (__int64)v88;
        v13 = v89;
        v19 = v16 - 1;
      }
      else if ( v18 == 80 )
      {
        if ( !v15 )
          v15 = v16;
        v21 = v15;
        if ( v15 > v13 )
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
        v18 = v16 - v15;
        v15 = 0;
        sub_2241130(&v88, v21, 0, "\\l...", 5);
        v19 = v16 + 3;
        v16 += 4;
        v14 = (__int64)v88;
        v13 = v89;
      }
      else
      {
        v19 = v16;
        ++v18;
        ++v16;
      }
      v17 = v16;
      if ( *(_BYTE *)(v14 + v19) == 32 )
        v15 = v19;
    }
    while ( v16 != v13 );
    v9 = v81;
    v2 = a2;
  }
LABEL_34:
  sub_16E7BC0(&v94);
  if ( v91 != (char *)v93 )
    j_j___libc_free_0(v91, v93[0] + 1LL);
LABEL_36:
  sub_16BE9B0(&v94, &v88);
  sub_16E7EE0(v9, v94, v95);
  if ( v94 != (const char *)&v96 )
    j_j___libc_free_0(v94, &v96->m128i_i8[1]);
  if ( v88 != v90 )
    j_j___libc_free_0(v88, v90[0] + 1LL);
  v28 = *(_QWORD *)v2;
  v29 = 0;
  v94 = (const char *)&unk_49EFBE0;
  v99 = (const char **)v86;
  v86[0] = v87;
  v86[1] = 0;
  LOBYTE(v87[0]) = 0;
  v98 = 1;
  v97 = 0;
  v96 = 0;
  v95 = 0;
  v30 = sub_157EBA0(v28 & 0xFFFFFFFFFFFFFFF8LL);
  for ( i = v30; ; i = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v33 = 0;
    if ( i )
      v33 = sub_15F4D60(i);
    if ( v33 == v29 )
      break;
    v32 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 32LL);
    if ( v32 != sub_15F4DF0(v30, v29) )
      break;
    ++v29;
  }
  v83 = 0;
  v34 = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v34 )
    v83 = sub_15F4D60(v34);
  if ( v83 != v29 )
  {
    v82 = 0;
    v35 = 0;
    while ( 1 )
    {
      v88 = v90;
      sub_144C6E0((__int64 *)&v88, byte_3F871B3, (__int64)byte_3F871B3);
      if ( v89 )
      {
        v36 = v97;
        if ( v35 )
        {
          if ( v96 == v97 )
          {
            sub_16E7EE0(&v94, "|", 1);
            v36 = v97;
          }
          else
          {
            v97->m128i_i8[0] = 124;
            v36 = (__m128i *)&v97->m128i_i8[1];
            v97 = (__m128i *)((char *)v97 + 1);
          }
        }
        if ( (unsigned __int64)((char *)v96 - (char *)v36) <= 1 )
        {
          v37 = (const char **)sub_16E7EE0(&v94, "<s", 2);
        }
        else
        {
          v37 = &v94;
          v36->m128i_i16[0] = 29500;
          v97 = (__m128i *)((char *)v97 + 2);
        }
        v38 = sub_16E7A90(v37, v35);
        v39 = *(_BYTE **)(v38 + 24);
        if ( *(_BYTE **)(v38 + 16) == v39 )
        {
          v38 = sub_16E7EE0(v38, ">", 1);
        }
        else
        {
          *v39 = 62;
          ++*(_QWORD *)(v38 + 24);
        }
        sub_16BE9B0(&v91, &v88);
        sub_16E7EE0(v38, v91, v92);
        if ( v91 != (char *)v93 )
          j_j___libc_free_0(v91, v93[0] + 1LL);
        if ( v88 != v90 )
          j_j___libc_free_0(v88, v90[0] + 1LL);
        v82 = 1;
      }
      else if ( v88 != v90 )
      {
        j_j___libc_free_0(v88, v90[0] + 1LL);
      }
      ++v29;
      while ( 1 )
      {
        v41 = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
        v42 = 0;
        if ( v41 )
          v42 = sub_15F4D60(v41);
        if ( v42 == v29 )
          break;
        v40 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 32LL);
        if ( v40 != sub_15F4DF0(v30, v29) )
          break;
        ++v29;
      }
      if ( v83 == v29 )
        break;
      if ( ++v35 == 64 )
      {
        if ( !v82 )
          goto LABEL_70;
        v70 = v97;
        if ( (unsigned __int64)((char *)v96 - (char *)v97) <= 0x11 )
        {
          sub_16E7EE0(&v94, "|<s64>truncated...", 18);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
          v97[1].m128i_i16[0] = 11822;
          *v70 = si128;
          v97 = (__m128i *)((char *)v97 + 18);
        }
        goto LABEL_116;
      }
    }
    if ( !v82 )
      goto LABEL_70;
LABEL_116:
    v72 = *a1;
    v73 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v73 )
    {
      sub_16E7EE0(v72, "|", 1);
    }
    else
    {
      *v73 = 124;
      ++*(_QWORD *)(v72 + 24);
    }
    v74 = *a1;
    v75 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v75 )
    {
      v74 = sub_16E7EE0(*a1, "{", 1);
    }
    else
    {
      *v75 = 123;
      ++*(_QWORD *)(v74 + 24);
    }
    if ( v97 != v95 )
      sub_16E7BA0(&v94);
    v76 = sub_16E7EE0(v74, *v99, v99[1]);
    v77 = *(_BYTE **)(v76 + 24);
    if ( *(_BYTE **)(v76 + 16) == v77 )
    {
      sub_16E7EE0(v76, "}", 1);
    }
    else
    {
      *v77 = 125;
      ++*(_QWORD *)(v76 + 24);
    }
  }
LABEL_70:
  v43 = *a1;
  v44 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v44) <= 4 )
  {
    sub_16E7EE0(v43, "}\"];\n", 5);
  }
  else
  {
    *(_DWORD *)v44 = 995959421;
    *(_BYTE *)(v44 + 4) = 10;
    *(_QWORD *)(v43 + 24) += 5LL;
  }
  v45 = *(_QWORD *)v2;
  v91 = v2;
  v46 = 0;
  v47 = v45;
  v92 = sub_157EBA0(v45 & 0xFFFFFFFFFFFFFFF8LL);
  v48 = v92;
  while ( 1 )
  {
    v50 = sub_157EBA0(v47 & 0xFFFFFFFFFFFFFFF8LL);
    v51 = 0;
    if ( v50 )
      v51 = sub_15F4D60(v50);
    if ( v51 == v46 )
      break;
    v49 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 32LL);
    if ( v49 != sub_15F4DF0(v48, v46) )
      break;
    v47 = *(_QWORD *)v2;
    ++v46;
  }
  v85 = 0;
  v52 = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v52 )
    v85 = sub_15F4D60(v52);
  v53 = 0;
  if ( v85 != v46 )
  {
    while ( 1 )
    {
      v54 = sub_15F4DF0(v48, v46);
      sub_1444DB0(*((_QWORD **)v2 + 1), v54);
      LODWORD(v93[0]) = v46;
      sub_144F340(a1, (__int64 *)v2, v53, v55, v56, v57, (__int64)v91, v92, v46);
      do
      {
        ++v46;
        v59 = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
        v60 = 0;
        if ( v59 )
          v60 = sub_15F4D60(v59);
        if ( v46 == v60 )
          break;
        v58 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 32LL);
      }
      while ( v58 == sub_15F4DF0(v48, v46) );
      ++v53;
      if ( v85 == v46 )
        break;
      if ( v53 == 64 )
      {
        while ( v85 != v46 )
        {
          v61 = sub_15F4DF0(v48, v46);
          sub_1444DB0(*((_QWORD **)v2 + 1), v61);
          LODWORD(v93[0]) = v46;
          v79 = v46++;
          sub_144F340(a1, (__int64 *)v2, 64, v62, v63, v64, (__int64)v91, v92, v79);
          while ( 1 )
          {
            v66 = sub_157EBA0(*(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL);
            v67 = 0;
            if ( v66 )
              v67 = sub_15F4D60(v66);
            if ( v67 == v46 )
              break;
            v65 = *(_QWORD *)(*((_QWORD *)v2 + 1) + 32LL);
            if ( v65 != sub_15F4DF0(v48, v46) )
              break;
            ++v46;
          }
        }
        break;
      }
    }
  }
  result = sub_16E7BC0(&v94);
  if ( (_QWORD *)v86[0] != v87 )
    return j_j___libc_free_0(v86[0], v87[0] + 1LL);
  return result;
}
