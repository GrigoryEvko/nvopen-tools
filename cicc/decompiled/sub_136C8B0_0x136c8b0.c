// Function: sub_136C8B0
// Address: 0x136c8b0
//
__int64 __fastcall sub_136C8B0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // r14
  unsigned int v4; // ebx
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rax
  void *v8; // rdx
  __int64 v9; // rdi
  _QWORD *v10; // rdx
  __int64 v11; // r15
  int v12; // ebx
  __int64 v13; // rax
  size_t v14; // rdx
  _BYTE *v15; // rdi
  const char *v16; // rsi
  unsigned __int64 v17; // rax
  const char **v18; // r8
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rbx
  char v22; // r14
  const char **v23; // rdi
  __int64 v24; // r14
  _BYTE *v25; // rax
  _WORD *v26; // rdx
  __int64 v27; // rdi
  _BYTE *v28; // rax
  __int64 v29; // r12
  _BYTE *v30; // rax
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  unsigned __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rdi
  unsigned int v38; // r14d
  int v39; // edx
  __int64 result; // rax
  _OWORD *v41; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // r12
  unsigned int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rdi
  _BYTE *v49; // rax
  __int64 v50; // r8
  _BYTE *v51; // rax
  __int64 v52; // r8
  _BYTE *v53; // rax
  _WORD *v54; // rdx
  __int64 v55; // r15
  __int64 i; // r12
  __int64 v57; // rsi
  unsigned __int64 v58; // rax
  int v59; // [rsp+Ch] [rbp-124h]
  size_t v60; // [rsp+20h] [rbp-110h]
  __int64 *v61; // [rsp+38h] [rbp-F8h]
  __int64 v62; // [rsp+38h] [rbp-F8h]
  __int64 v63; // [rsp+38h] [rbp-F8h]
  __int64 v64; // [rsp+38h] [rbp-F8h]
  int v65; // [rsp+40h] [rbp-F0h]
  const char *v67; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v68; // [rsp+58h] [rbp-D8h]
  _QWORD v69[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v70[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v71[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v72; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v73; // [rsp+98h] [rbp-98h]
  _QWORD v74[2]; // [rsp+A0h] [rbp-90h] BYREF
  const char *v75; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v76; // [rsp+B8h] [rbp-78h]
  _QWORD v77[2]; // [rsp+C0h] [rbp-70h] BYREF
  const char *v78; // [rsp+D0h] [rbp-60h] BYREF
  void *v79; // [rsp+D8h] [rbp-58h]
  _BYTE *v80; // [rsp+E0h] [rbp-50h] BYREF
  void *dest; // [rsp+E8h] [rbp-48h]
  int v82; // [rsp+F0h] [rbp-40h]
  const char **v83; // [rsp+F8h] [rbp-38h]

  v3 = *(__int64 **)a1[1];
  v68 = 0;
  LOBYTE(v69[0]) = 0;
  v4 = qword_4F982C0[20];
  v67 = (const char *)v69;
  if ( v4 )
  {
    if ( !a1[3] )
    {
      v55 = *(_QWORD *)(sub_1368BD0(v3) + 80);
      for ( i = sub_1368BD0(v3) + 72; i != v55; v55 = *(_QWORD *)(v55 + 8) )
      {
        v57 = v55 - 24;
        if ( !v55 )
          v57 = 0;
        v58 = sub_1368AA0(v3, v57);
        if ( v58 < a1[3] )
          v58 = a1[3];
        a1[3] = v58;
      }
    }
    v45 = sub_1368AA0(v3, a2);
    v46 = sub_16AF730(v4, 100);
    v78 = (const char *)a1[3];
    if ( sub_16AF500(&v78, v46) <= v45 )
    {
      v82 = 1;
      dest = 0;
      v78 = (const char *)&unk_49EFBE0;
      v80 = 0;
      v79 = 0;
      v83 = &v67;
      sub_16E7EE0(&v78, "color=\"red\"", 11);
      if ( dest != v79 )
        sub_16E7BA0(&v78);
      sub_16E7BC0(&v78);
    }
  }
  v5 = *a1;
  v6 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v6) <= 4 )
  {
    v5 = sub_16E7EE0(v5, "\tNode", 5);
  }
  else
  {
    *(_DWORD *)v6 = 1685016073;
    *(_BYTE *)(v6 + 4) = 101;
    *(_QWORD *)(v5 + 24) += 5LL;
  }
  v7 = sub_16E7B40(v5, a2);
  v8 = *(void **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0xEu )
  {
    sub_16E7EE0(v7, " [shape=record,", 15);
  }
  else
  {
    qmemcpy(v8, " [shape=record,", 15);
    *(_QWORD *)(v7 + 24) += 15LL;
  }
  if ( v68 )
  {
    v48 = sub_16E7EE0(*a1, v67);
    v49 = *(_BYTE **)(v48 + 24);
    if ( *(_BYTE **)(v48 + 16) == v49 )
    {
      sub_16E7EE0(v48, ",", 1);
    }
    else
    {
      *v49 = 44;
      ++*(_QWORD *)(v48 + 24);
    }
  }
  v9 = *a1;
  v10 = *(_QWORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v10 <= 7u )
  {
    sub_16E7EE0(v9, "label=\"{", 8);
  }
  else
  {
    *v10 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v9 + 24) += 8LL;
  }
  v11 = *a1;
  v61 = *(__int64 **)a1[1];
  if ( unk_4F98100 == 1 )
    v12 = 3;
  else
    v12 = dword_4F98560;
  LOBYTE(v77[0]) = 0;
  v75 = (const char *)v77;
  v76 = 0;
  v82 = 1;
  dest = 0;
  v80 = 0;
  v79 = 0;
  v78 = (const char *)&unk_49EFBE0;
  v83 = &v75;
  v13 = sub_1649960(a2);
  v15 = dest;
  v16 = (const char *)v13;
  v17 = v80 - (_BYTE *)dest;
  if ( v80 - (_BYTE *)dest < v14 )
  {
    v47 = sub_16E7EE0(&v78, v16);
    v15 = *(_BYTE **)(v47 + 24);
    v18 = (const char **)v47;
    v17 = *(_QWORD *)(v47 + 16) - (_QWORD)v15;
LABEL_13:
    if ( v17 > 2 )
      goto LABEL_14;
    goto LABEL_76;
  }
  v18 = &v78;
  if ( !v14 )
    goto LABEL_13;
  v60 = v14;
  memcpy(dest, v16, v14);
  v18 = &v78;
  v43 = v80 - ((_BYTE *)dest + v60);
  dest = (char *)dest + v60;
  v15 = dest;
  if ( v43 > 2 )
  {
LABEL_14:
    v15[2] = 32;
    *(_WORD *)v15 = 14880;
    v18[3] += 3;
    if ( v12 != 2 )
      goto LABEL_15;
    goto LABEL_77;
  }
LABEL_76:
  sub_16E7EE0(v18, " : ", 3);
  if ( v12 != 2 )
  {
LABEL_15:
    if ( v12 == 3 )
    {
      sub_1368C40((__int64)&v72, v61, a2);
      if ( (_BYTE)v73 )
      {
        sub_16E7A90(&v78, v72);
      }
      else
      {
        v54 = dest;
        if ( (unsigned __int64)(v80 - (_BYTE *)dest) <= 6 )
        {
          sub_16E7EE0(&v78, "Unknown", 7);
        }
        else
        {
          *(_DWORD *)dest = 1852534357;
          v54[2] = 30575;
          *((_BYTE *)v54 + 6) = 110;
          dest = (char *)dest + 7;
        }
      }
    }
    else if ( v12 == 1 )
    {
      sub_1368D20(v61, (__int64)&v78, a2);
    }
    goto LABEL_18;
  }
LABEL_77:
  v44 = sub_1368AA0(v61, a2);
  sub_16E7A90(&v78, v44);
LABEL_18:
  sub_16E7BC0(&v78);
  sub_16BE9B0(&v78, &v75);
  sub_16E7EE0(v11, v78, v79);
  if ( v78 != (const char *)&v80 )
    j_j___libc_free_0(v78, v80 + 1);
  if ( v75 != (const char *)v77 )
    j_j___libc_free_0(v75, v77[0] + 1LL);
  v72 = v74;
  sub_1367D20((__int64 *)&v72, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v73 )
  {
    v52 = *a1;
    v53 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v53 )
    {
      v52 = sub_16E7EE0(*a1, "|", 1);
    }
    else
    {
      *v53 = 124;
      ++*(_QWORD *)(v52 + 24);
    }
    v64 = v52;
    sub_16BE9B0(&v78, &v72);
    sub_16E7EE0(v64, v78, v79);
    if ( v78 != (const char *)&v80 )
      j_j___libc_free_0(v78, v80 + 1);
  }
  v75 = (const char *)v77;
  sub_1367D20((__int64 *)&v75, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v76 )
  {
    v50 = *a1;
    v51 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v51 )
    {
      v50 = sub_16E7EE0(*a1, "|", 1);
    }
    else
    {
      *v51 = 124;
      ++*(_QWORD *)(v50 + 24);
    }
    v63 = v50;
    sub_16BE9B0(&v78, &v75);
    sub_16E7EE0(v63, v78, v79);
    if ( v78 != (const char *)&v80 )
      j_j___libc_free_0(v78, v80 + 1);
  }
  if ( v75 != (const char *)v77 )
    j_j___libc_free_0(v75, v77[0] + 1LL);
  if ( v72 != v74 )
    j_j___libc_free_0(v72, v74[0] + 1LL);
  v70[1] = 0;
  v70[0] = v71;
  LOBYTE(v71[0]) = 0;
  v82 = 1;
  dest = 0;
  v80 = 0;
  v79 = 0;
  v78 = (const char *)&unk_49EFBE0;
  v83 = (const char **)v70;
  v19 = sub_157EBA0(a2);
  if ( !v19 )
    goto LABEL_58;
  v20 = sub_15F4D60(v19);
  v59 = v20;
  if ( !v20 )
    goto LABEL_58;
  v21 = 0;
  v22 = 0;
  v62 = (unsigned int)(v20 - 1);
  do
  {
    v72 = v74;
    sub_1367D20((__int64 *)&v72, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v73 )
    {
      v26 = dest;
      if ( (_DWORD)v21 )
      {
        if ( v80 != dest )
        {
          *(_BYTE *)dest = 124;
          v26 = (char *)dest + 1;
          dest = v26;
          if ( (unsigned __int64)(v80 - (_BYTE *)v26) <= 1 )
            goto LABEL_46;
          goto LABEL_32;
        }
        sub_16E7EE0(&v78, "|", 1);
        v26 = dest;
      }
      if ( (unsigned __int64)(v80 - (_BYTE *)v26) <= 1 )
      {
LABEL_46:
        v23 = (const char **)sub_16E7EE0(&v78, "<s", 2);
        goto LABEL_33;
      }
LABEL_32:
      v23 = &v78;
      *v26 = 29500;
      dest = (char *)dest + 2;
LABEL_33:
      v24 = sub_16E7A90(v23, v21);
      v25 = *(_BYTE **)(v24 + 24);
      if ( *(_BYTE **)(v24 + 16) == v25 )
      {
        v24 = sub_16E7EE0(v24, ">", 1);
      }
      else
      {
        *v25 = 62;
        ++*(_QWORD *)(v24 + 24);
      }
      sub_16BE9B0(&v75, &v72);
      sub_16E7EE0(v24, v75, v76);
      if ( v75 != (const char *)v77 )
        j_j___libc_free_0(v75, v77[0] + 1LL);
      if ( v72 != v74 )
        j_j___libc_free_0(v72, v74[0] + 1LL);
      v22 = 1;
LABEL_40:
      if ( v62 == v21 )
        goto LABEL_50;
      goto LABEL_41;
    }
    if ( v72 == v74 )
      goto LABEL_40;
    j_j___libc_free_0(v72, v74[0] + 1LL);
    if ( v62 == v21 )
      goto LABEL_50;
LABEL_41:
    ++v21;
  }
  while ( v21 != 64 );
  if ( v59 == 64 )
  {
LABEL_50:
    if ( !v22 )
      goto LABEL_58;
    goto LABEL_51;
  }
  if ( !v22 )
    goto LABEL_58;
  v41 = dest;
  if ( (unsigned __int64)(v80 - (_BYTE *)dest) <= 0x11 )
  {
    sub_16E7EE0(&v78, "|<s64>truncated...", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
    *((_WORD *)dest + 8) = 11822;
    *v41 = si128;
    dest = (char *)dest + 18;
  }
LABEL_51:
  v27 = *a1;
  v28 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v28 )
  {
    sub_16E7EE0(v27, "|", 1);
  }
  else
  {
    *v28 = 124;
    ++*(_QWORD *)(v27 + 24);
  }
  v29 = *a1;
  v30 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v30 )
  {
    v29 = sub_16E7EE0(*a1, "{", 1);
    if ( dest != v79 )
LABEL_55:
      sub_16E7BA0(&v78);
  }
  else
  {
    *v30 = 123;
    ++*(_QWORD *)(v29 + 24);
    if ( dest != v79 )
      goto LABEL_55;
  }
  v31 = sub_16E7EE0(v29, *v83, v83[1]);
  v32 = *(_BYTE **)(v31 + 24);
  if ( *(_BYTE **)(v31 + 16) == v32 )
  {
    sub_16E7EE0(v31, "}", 1);
  }
  else
  {
    *v32 = 125;
    ++*(_QWORD *)(v31 + 24);
  }
LABEL_58:
  v33 = *a1;
  v34 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v34) <= 4 )
  {
    sub_16E7EE0(v33, "}\"];\n", 5);
  }
  else
  {
    *(_DWORD *)v34 = 995959421;
    *(_BYTE *)(v34 + 4) = 10;
    *(_QWORD *)(v33 + 24) += 5LL;
  }
  v35 = 0;
  v36 = sub_157EBA0(a2);
  v37 = sub_157EBA0(a2);
  if ( v37 )
  {
    v65 = sub_15F4D60(v37);
    if ( v65 )
    {
      v38 = 0;
      while ( 1 )
      {
        sub_15F4DF0(v36, v38);
        v39 = v38;
        v35 = v38++ | v35 & 0xFFFFFFFF00000000LL;
        sub_136C3D0((__int64)a1, a2, v39, v36, v35);
        if ( v38 == v65 )
          break;
        if ( v38 == 64 )
        {
          if ( v65 != 64 )
          {
            do
            {
              sub_15F4DF0(v36, v38);
              v35 = v38++ | v35 & 0xFFFFFFFF00000000LL;
              sub_136C3D0((__int64)a1, a2, 64, v36, v35);
            }
            while ( v65 != v38 );
          }
          break;
        }
      }
    }
  }
  result = sub_16E7BC0(&v78);
  if ( (_QWORD *)v70[0] != v71 )
    result = j_j___libc_free_0(v70[0], v71[0] + 1LL);
  if ( v67 != (const char *)v69 )
    return j_j___libc_free_0(v67, v69[0] + 1LL);
  return result;
}
