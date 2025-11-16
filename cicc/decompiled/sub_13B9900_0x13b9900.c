// Function: sub_13B9900
// Address: 0x13b9900
//
__int64 __fastcall sub_13B9900(_BYTE *a1, __int64 *a2)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rax
  void *v5; // rdx
  __int64 v6; // rdi
  _QWORD *v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rbx
  char v11; // r15
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // r15
  _BYTE *v16; // rax
  char *v17; // rdx
  __int64 v18; // rdx
  const char *v19; // rdx
  _BYTE *v20; // rbx
  size_t v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdx
  int v26; // r12d
  __int64 *v27; // rbx
  __int64 v28; // rdi
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  _WORD *v31; // rdx
  __int64 v32; // r14
  __int64 v33; // r15
  bool v34; // zf
  int v35; // r15d
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rdi
  _WORD *v39; // rdx
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __int64 v42; // rdi
  _BYTE *v43; // rax
  __int64 result; // rax
  __m128i *v45; // rdx
  __m128i v46; // xmm0
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __int64 v49; // r12
  _BYTE *v50; // rax
  __int64 v51; // rdi
  _BYTE *v52; // rax
  __int64 v53; // rdi
  _QWORD *v54; // rdx
  __int64 v55; // rdi
  _WORD *v56; // rdx
  __int64 v57; // r15
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdi
  _BYTE *v62; // rax
  __int64 v63; // rdi
  _BYTE *v64; // rax
  __int64 v65; // rax
  __m128i si128; // xmm0
  _QWORD *v67; // rdi
  __int64 v68; // [rsp+30h] [rbp-F0h]
  __int64 *v69; // [rsp+30h] [rbp-F0h]
  _QWORD v71[2]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD v72[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v73; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v74; // [rsp+88h] [rbp-98h]
  _QWORD v75[2]; // [rsp+90h] [rbp-90h] BYREF
  const char *v76; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-78h]
  _QWORD v78[2]; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v79; // [rsp+C0h] [rbp-60h] BYREF
  char *v80; // [rsp+C8h] [rbp-58h]
  char *v81; // [rsp+D0h] [rbp-50h] BYREF
  char *v82; // [rsp+D8h] [rbp-48h]
  int v83; // [rsp+E0h] [rbp-40h]
  const char **v84; // [rsp+E8h] [rbp-38h]

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(v2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v2 + 16) - v3) <= 4 )
  {
    v2 = sub_16E7EE0(v2, "\tNode", 5);
  }
  else
  {
    *(_DWORD *)v3 = 1685016073;
    *(_BYTE *)(v3 + 4) = 101;
    *(_QWORD *)(v2 + 24) += 5LL;
  }
  v4 = sub_16E7B40(v2, a2);
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xEu )
  {
    sub_16E7EE0(v4, " [shape=record,", 15);
  }
  else
  {
    qmemcpy(v5, " [shape=record,", 15);
    *(_QWORD *)(v4 + 24) += 15LL;
  }
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v7 <= 7u )
  {
    sub_16E7EE0(v6, "label=\"{", 8);
  }
  else
  {
    *v7 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v6 + 24) += 8LL;
  }
  v8 = *(_QWORD *)a1;
  v9 = *a2;
  if ( *a2 )
  {
    if ( !a1[16] )
    {
      sub_13B65F0((__int64 *)&v73, *a2);
LABEL_10:
      sub_16BE9B0(&v79, &v73);
      goto LABEL_11;
    }
    sub_1649960(*a2);
    if ( !v18 )
    {
      LOBYTE(v78[0]) = 0;
      v84 = &v76;
      v79 = (__int64)&unk_49EFBE0;
      v76 = (const char *)v78;
      v77 = 0;
      v83 = 1;
      v82 = 0;
      v81 = 0;
      v80 = 0;
      sub_15537D0(v9, &v79, 0);
      if ( v82 != v80 )
        sub_16E7BA0(&v79);
      v73 = v75;
      sub_13B5790((__int64 *)&v73, *v84, (__int64)&v84[1][(_QWORD)*v84]);
      sub_16E7BC0(&v79);
      if ( v76 != (const char *)v78 )
        j_j___libc_free_0(v76, v78[0] + 1LL);
      goto LABEL_10;
    }
    v20 = (_BYTE *)sub_1649960(v9);
    v21 = (size_t)v19;
    if ( v20 )
    {
      v79 = (__int64)v19;
      v22 = (__int64)v19;
      v73 = v75;
      if ( (unsigned __int64)v19 > 0xF )
      {
        v73 = (_QWORD *)sub_22409D0(&v73, &v79, 0);
        v67 = v73;
        v75[0] = v79;
      }
      else
      {
        if ( v19 == (const char *)1 )
        {
          LOBYTE(v75[0]) = *v20;
          v23 = v75;
LABEL_38:
          v74 = v22;
          *((_BYTE *)v23 + v22) = 0;
          goto LABEL_10;
        }
        if ( !v19 )
        {
          v23 = v75;
          goto LABEL_38;
        }
        v67 = v75;
      }
      memcpy(v67, v20, v21);
      v22 = v79;
      v23 = v73;
      goto LABEL_38;
    }
    v74 = 0;
    v73 = v75;
    LOBYTE(v75[0]) = 0;
    sub_16BE9B0(&v79, &v73);
  }
  else
  {
    v79 = 24;
    v73 = v75;
    v65 = sub_22409D0(&v73, &v79, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_4289C20);
    v73 = (_QWORD *)v65;
    v75[0] = v79;
    *(_QWORD *)(v65 + 16) = 0x65646F6E20746F6FLL;
    *(__m128i *)v65 = si128;
    v74 = v79;
    *((_BYTE *)v73 + v79) = 0;
    sub_16BE9B0(&v79, &v73);
  }
LABEL_11:
  sub_16E7EE0(v8, (const char *)v79, v80);
  if ( (char **)v79 != &v81 )
    j_j___libc_free_0(v79, v81 + 1);
  if ( v73 != v75 )
    j_j___libc_free_0(v73, v75[0] + 1LL);
  v10 = 0;
  v71[1] = 0;
  v84 = (const char **)v71;
  v11 = 0;
  v12 = a2[3];
  v13 = a2[4];
  v71[0] = v72;
  LOBYTE(v72[0]) = 0;
  v83 = 1;
  v82 = 0;
  v81 = 0;
  v80 = 0;
  v79 = (__int64)&unk_49EFBE0;
  v68 = v13;
  if ( v12 == v13 )
    goto LABEL_43;
  do
  {
    v73 = v75;
    sub_13B5840((__int64 *)&v73, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v74 )
    {
      v17 = v82;
      if ( v10 )
      {
        if ( v81 != v82 )
        {
          *v82 = 124;
          v17 = v82 + 1;
          v82 = v17;
          if ( (unsigned __int64)(v81 - v17) <= 1 )
            goto LABEL_32;
          goto LABEL_18;
        }
        sub_16E7EE0(&v79, "|", 1);
        v17 = v82;
      }
      if ( (unsigned __int64)(v81 - v17) <= 1 )
      {
LABEL_32:
        v14 = (__int64 *)sub_16E7EE0(&v79, "<s", 2);
        goto LABEL_19;
      }
LABEL_18:
      v14 = &v79;
      *(_WORD *)v17 = 29500;
      v82 += 2;
LABEL_19:
      v15 = sub_16E7A90(v14, v10);
      v16 = *(_BYTE **)(v15 + 24);
      if ( *(_BYTE **)(v15 + 16) == v16 )
      {
        v15 = sub_16E7EE0(v15, ">", 1);
      }
      else
      {
        *v16 = 62;
        ++*(_QWORD *)(v15 + 24);
      }
      sub_16BE9B0(&v76, &v73);
      sub_16E7EE0(v15, v76, v77);
      if ( v76 != (const char *)v78 )
        j_j___libc_free_0(v76, v78[0] + 1LL);
      if ( v73 != v75 )
        j_j___libc_free_0(v73, v75[0] + 1LL);
      v11 = 1;
LABEL_26:
      v12 += 8;
      if ( v68 == v12 )
        goto LABEL_42;
      goto LABEL_27;
    }
    if ( v73 == v75 )
      goto LABEL_26;
    v12 += 8;
    j_j___libc_free_0(v73, v75[0] + 1LL);
    if ( v68 == v12 )
    {
LABEL_42:
      if ( !v11 )
        goto LABEL_43;
      goto LABEL_81;
    }
LABEL_27:
    ++v10;
  }
  while ( v10 != 64 );
  if ( !v11 )
    goto LABEL_43;
  v45 = (__m128i *)v82;
  if ( (unsigned __int64)(v81 - v82) <= 0x11 )
  {
    sub_16E7EE0(&v79, "|<s64>truncated...", 18);
  }
  else
  {
    v46 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
    *((_WORD *)v82 + 8) = 11822;
    *v45 = v46;
    v82 += 18;
  }
LABEL_81:
  v47 = *(_QWORD *)a1;
  v48 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v48 )
  {
    sub_16E7EE0(v47, "|", 1);
  }
  else
  {
    *v48 = 124;
    ++*(_QWORD *)(v47 + 24);
  }
  v49 = *(_QWORD *)a1;
  v50 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v50 )
  {
    v49 = sub_16E7EE0(*(_QWORD *)a1, "{", 1);
    if ( v82 != v80 )
LABEL_85:
      sub_16E7BA0(&v79);
  }
  else
  {
    *v50 = 123;
    ++*(_QWORD *)(v49 + 24);
    if ( v82 != v80 )
      goto LABEL_85;
  }
  v51 = sub_16E7EE0(v49, *v84, v84[1]);
  v52 = *(_BYTE **)(v51 + 24);
  if ( *(_BYTE **)(v51 + 16) == v52 )
  {
    sub_16E7EE0(v51, "}", 1);
  }
  else
  {
    *v52 = 125;
    ++*(_QWORD *)(v51 + 24);
  }
LABEL_43:
  v24 = *(_QWORD *)a1;
  v25 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v25) <= 4 )
  {
    sub_16E7EE0(v24, "}\"];\n", 5);
  }
  else
  {
    *(_DWORD *)v25 = 995959421;
    *(_BYTE *)(v25 + 4) = 10;
    *(_QWORD *)(v24 + 24) += 5LL;
  }
  v26 = 0;
  v27 = (__int64 *)a2[3];
  v69 = (__int64 *)a2[4];
  if ( v69 != v27 )
  {
    do
    {
      v32 = *v27;
      if ( *v27 )
      {
        v76 = (const char *)v78;
        sub_13B5840((__int64 *)&v76, byte_3F871B3, (__int64)byte_3F871B3);
        v33 = v77;
        if ( v76 != (const char *)v78 )
          j_j___libc_free_0(v76, v78[0] + 1LL);
        v34 = v33 == 0;
        v35 = -1;
        if ( !v34 )
          v35 = v26;
        v76 = (const char *)v78;
        sub_13B5840((__int64 *)&v76, byte_3F871B3, (__int64)byte_3F871B3);
        v36 = *(_QWORD *)a1;
        v37 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v37) > 4 )
        {
          *(_DWORD *)v37 = 1685016073;
          *(_BYTE *)(v37 + 4) = 101;
          *(_QWORD *)(v36 + 24) += 5LL;
        }
        else
        {
          v36 = sub_16E7EE0(v36, "\tNode", 5);
        }
        sub_16E7B40(v36, a2);
        if ( v35 != -1 )
        {
          v38 = *(_QWORD *)a1;
          v39 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v39 <= 1u )
          {
            v38 = sub_16E7EE0(v38, ":s", 2);
          }
          else
          {
            *v39 = 29498;
            *(_QWORD *)(v38 + 24) += 2LL;
          }
          sub_16E7AB0(v38, v35);
        }
        v28 = *(_QWORD *)a1;
        v29 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v29 <= 7u )
        {
          v28 = sub_16E7EE0(v28, " -> Node", 8);
        }
        else
        {
          *v29 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v28 + 24) += 8LL;
        }
        sub_16E7B40(v28, v32);
        if ( v77 )
        {
          v40 = *(_QWORD *)a1;
          v41 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v41 )
          {
            v40 = sub_16E7EE0(v40, "[", 1);
          }
          else
          {
            *v41 = 91;
            ++*(_QWORD *)(v40 + 24);
          }
          v42 = sub_16E7EE0(v40, v76, v77);
          v43 = *(_BYTE **)(v42 + 24);
          if ( *(_BYTE **)(v42 + 16) == v43 )
          {
            sub_16E7EE0(v42, "]", 1);
          }
          else
          {
            *v43 = 93;
            ++*(_QWORD *)(v42 + 24);
          }
        }
        v30 = *(_QWORD *)a1;
        v31 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v31 <= 1u )
        {
          sub_16E7EE0(v30, ";\n", 2);
        }
        else
        {
          *v31 = 2619;
          *(_QWORD *)(v30 + 24) += 2LL;
        }
        if ( v76 != (const char *)v78 )
          j_j___libc_free_0(v76, v78[0] + 1LL);
      }
      ++v27;
      ++v26;
      if ( v69 == v27 )
        goto LABEL_75;
    }
    while ( v26 != 64 );
    while ( 1 )
    {
      v57 = *v27;
      if ( *v27 )
        break;
LABEL_109:
      if ( v69 == ++v27 )
        goto LABEL_75;
    }
    v58 = *(_QWORD *)a1;
    v77 = 0;
    v76 = (const char *)v78;
    LOBYTE(v78[0]) = 0;
    v59 = *(_QWORD *)(v58 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v58 + 16) - v59) > 4 )
    {
      *(_DWORD *)v59 = 1685016073;
      *(_BYTE *)(v59 + 4) = 101;
      *(_QWORD *)(v58 + 24) += 5LL;
    }
    else
    {
      v58 = sub_16E7EE0(v58, "\tNode", 5);
    }
    sub_16E7B40(v58, a2);
    v53 = *(_QWORD *)a1;
    v54 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v54 <= 7u )
    {
      v60 = sub_16E7EE0(v53, " -> Node", 8);
      sub_16E7B40(v60, v57);
      if ( !v77 )
        goto LABEL_105;
    }
    else
    {
      *v54 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v53 + 24) += 8LL;
      sub_16E7B40(v53, v57);
      if ( !v77 )
        goto LABEL_105;
    }
    v61 = *(_QWORD *)a1;
    v62 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v62 )
    {
      v61 = sub_16E7EE0(v61, "[", 1);
    }
    else
    {
      *v62 = 91;
      ++*(_QWORD *)(v61 + 24);
    }
    v63 = sub_16E7EE0(v61, v76, v77);
    v64 = *(_BYTE **)(v63 + 24);
    if ( *(_BYTE **)(v63 + 16) == v64 )
    {
      sub_16E7EE0(v63, "]", 1);
    }
    else
    {
      *v64 = 93;
      ++*(_QWORD *)(v63 + 24);
    }
LABEL_105:
    v55 = *(_QWORD *)a1;
    v56 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v56 <= 1u )
    {
      sub_16E7EE0(v55, ";\n", 2);
    }
    else
    {
      *v56 = 2619;
      *(_QWORD *)(v55 + 24) += 2LL;
    }
    if ( v76 != (const char *)v78 )
      j_j___libc_free_0(v76, v78[0] + 1LL);
    goto LABEL_109;
  }
LABEL_75:
  result = sub_16E7BC0(&v79);
  if ( (_QWORD *)v71[0] != v72 )
    return j_j___libc_free_0(v71[0], v72[0] + 1LL);
  return result;
}
