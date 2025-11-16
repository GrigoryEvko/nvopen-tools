// Function: sub_139A6D0
// Address: 0x139a6d0
//
__int64 __fastcall sub_139A6D0(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  void *v6; // rdx
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  __int64 v12; // r12
  char v13; // r15
  __int64 v14; // r13
  __int64 v15; // rax
  const char **v16; // rdi
  __int64 v17; // r15
  _BYTE *v18; // rax
  char *v19; // rdx
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // r12
  _BYTE *v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  int v28; // r14d
  _QWORD *v29; // r12
  bool v30; // r15
  __int64 v31; // r13
  __int64 v32; // r15
  bool v33; // zf
  int v34; // r15d
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rdi
  _QWORD *v38; // rdx
  __int64 v39; // rdi
  _WORD *v40; // rdx
  __int64 v41; // rdi
  _WORD *v42; // rdx
  __int64 v43; // rdi
  _BYTE *v44; // rax
  __int64 v45; // rdi
  _BYTE *v46; // rax
  _QWORD *v47; // r14
  _QWORD *v48; // rcx
  __int64 result; // rax
  __m128i *v50; // rdx
  __m128i si128; // xmm0
  __int64 v52; // r13
  _BYTE *v53; // rax
  __int64 v54; // r13
  _BYTE *v55; // rax
  __int64 v56; // rdi
  _BYTE *v57; // rax
  __int64 v59; // [rsp+38h] [rbp-F8h]
  _QWORD *v60; // [rsp+38h] [rbp-F8h]
  const char *v62[2]; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD v63[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v64[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v65[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v66; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+98h] [rbp-98h]
  _QWORD v68[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v69; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v70; // [rsp+B8h] [rbp-78h]
  __int64 v71; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v72; // [rsp+C8h] [rbp-68h]
  char **v73; // [rsp+D0h] [rbp-60h] BYREF
  char *v74; // [rsp+D8h] [rbp-58h]
  char *v75; // [rsp+E0h] [rbp-50h] BYREF
  char *v76; // [rsp+E8h] [rbp-48h]
  int v77; // [rsp+F0h] [rbp-40h]
  const char **v78; // [rsp+F8h] [rbp-38h]

  v62[0] = (const char *)v63;
  sub_1399600((__int64 *)v62, byte_3F871B3, (__int64)byte_3F871B3);
  v3 = *a1;
  v4 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v4) <= 4 )
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
  if ( v62[1] )
  {
    v56 = sub_16E7EE0(*a1, v62[0]);
    v57 = *(_BYTE **)(v56 + 24);
    if ( *(_BYTE **)(v56 + 16) == v57 )
    {
      sub_16E7EE0(v56, ",", 1);
    }
    else
    {
      *v57 = 44;
      ++*(_QWORD *)(v56 + 24);
    }
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
  if ( *a2 )
  {
    v11 = (_BYTE *)sub_1649960(*a2);
    if ( v11 )
    {
      v69 = (__int64)&v71;
      sub_1399600(&v69, v11, (__int64)&v11[v10]);
    }
    else
    {
      v70 = 0;
      v69 = (__int64)&v71;
      LOBYTE(v71) = 0;
    }
  }
  else
  {
    v69 = (__int64)&v71;
    sub_1399600(&v69, "external node", (__int64)"");
  }
  sub_16BE9B0(&v73, &v69);
  sub_16E7EE0(v9, (const char *)v73, v74);
  if ( v73 != &v75 )
    j_j___libc_free_0(v73, v75 + 1);
  if ( (__int64 *)v69 != &v71 )
    j_j___libc_free_0(v69, v71 + 1);
  v66 = v68;
  sub_1399600((__int64 *)&v66, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v67 )
  {
    v54 = *a1;
    v55 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v55 )
    {
      v54 = sub_16E7EE0(*a1, "|", 1);
    }
    else
    {
      *v55 = 124;
      ++*(_QWORD *)(v54 + 24);
    }
    sub_16BE9B0(&v73, &v66);
    sub_16E7EE0(v54, (const char *)v73, v74);
    if ( v73 != &v75 )
      j_j___libc_free_0(v73, v75 + 1);
  }
  v69 = (__int64)&v71;
  sub_1399600(&v69, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v70 )
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
    sub_16BE9B0(&v73, &v69);
    sub_16E7EE0(v52, (const char *)v73, v74);
    if ( v73 != &v75 )
      j_j___libc_free_0(v73, v75 + 1);
  }
  if ( (__int64 *)v69 != &v71 )
    j_j___libc_free_0(v69, v71 + 1);
  if ( v66 != v68 )
    j_j___libc_free_0(v66, v68[0] + 1LL);
  v12 = 0;
  v78 = (const char **)v64;
  v13 = 0;
  v14 = a2[1];
  v15 = a2[2];
  v64[0] = v65;
  v64[1] = 0;
  LOBYTE(v65[0]) = 0;
  v77 = 1;
  v76 = 0;
  v75 = 0;
  v74 = 0;
  v73 = (char **)&unk_49EFBE0;
  v59 = v15;
  if ( v14 == v15 )
    goto LABEL_51;
  do
  {
    v66 = v68;
    sub_1399600((__int64 *)&v66, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v67 )
    {
      v19 = v76;
      if ( v12 )
      {
        if ( v76 != v75 )
        {
          *v76 = 124;
          v19 = v76 + 1;
          v76 = v19;
          if ( (unsigned __int64)(v75 - v19) <= 1 )
            goto LABEL_38;
          goto LABEL_24;
        }
        sub_16E7EE0(&v73, "|", 1);
        v19 = v76;
      }
      if ( (unsigned __int64)(v75 - v19) <= 1 )
      {
LABEL_38:
        v16 = (const char **)sub_16E7EE0(&v73, "<s", 2);
        goto LABEL_25;
      }
LABEL_24:
      *(_WORD *)v19 = 29500;
      v16 = (const char **)&v73;
      v76 += 2;
LABEL_25:
      v17 = sub_16E7A90(v16, v12);
      v18 = *(_BYTE **)(v17 + 24);
      if ( *(_BYTE **)(v17 + 16) == v18 )
      {
        v17 = sub_16E7EE0(v17, ">", 1);
      }
      else
      {
        *v18 = 62;
        ++*(_QWORD *)(v17 + 24);
      }
      sub_16BE9B0(&v69, &v66);
      sub_16E7EE0(v17, (const char *)v69, v70);
      if ( (__int64 *)v69 != &v71 )
        j_j___libc_free_0(v69, v71 + 1);
      if ( v66 != v68 )
        j_j___libc_free_0(v66, v68[0] + 1LL);
      v13 = 1;
LABEL_32:
      v14 += 32;
      if ( v59 == v14 )
        goto LABEL_43;
      goto LABEL_33;
    }
    if ( v66 == v68 )
      goto LABEL_32;
    v14 += 32;
    j_j___libc_free_0(v66, v68[0] + 1LL);
    if ( v59 == v14 )
    {
LABEL_43:
      if ( !v13 )
        goto LABEL_51;
      goto LABEL_44;
    }
LABEL_33:
    ++v12;
  }
  while ( v12 != 64 );
  if ( !v13 )
    goto LABEL_51;
  v50 = (__m128i *)v76;
  if ( (unsigned __int64)(v75 - v76) <= 0x11 )
  {
    sub_16E7EE0(&v73, "|<s64>truncated...", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
    *((_WORD *)v76 + 8) = 11822;
    *v50 = si128;
    v76 += 18;
  }
LABEL_44:
  v20 = *a1;
  v21 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v21 )
  {
    sub_16E7EE0(v20, "|", 1);
  }
  else
  {
    *v21 = 124;
    ++*(_QWORD *)(v20 + 24);
  }
  v22 = *a1;
  v23 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v23 )
  {
    v22 = sub_16E7EE0(*a1, "{", 1);
    if ( v76 != v74 )
LABEL_48:
      sub_16E7BA0(&v73);
  }
  else
  {
    *v23 = 123;
    ++*(_QWORD *)(v22 + 24);
    if ( v76 != v74 )
      goto LABEL_48;
  }
  v24 = sub_16E7EE0(v22, *v78, v78[1]);
  v25 = *(_BYTE **)(v24 + 24);
  if ( *(_BYTE **)(v24 + 16) == v25 )
  {
    sub_16E7EE0(v24, "}", 1);
  }
  else
  {
    *v25 = 125;
    ++*(_QWORD *)(v24 + 24);
  }
LABEL_51:
  v26 = *a1;
  v27 = *(_QWORD *)(*a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v27) <= 4 )
  {
    sub_16E7EE0(v26, "}\"];\n", 5);
  }
  else
  {
    *(_DWORD *)v27 = 995959421;
    *(_BYTE *)(v27 + 4) = 10;
    *(_QWORD *)(v26 + 24) += 5LL;
  }
  v28 = 0;
  v29 = (_QWORD *)a2[1];
  v60 = (_QWORD *)a2[2];
  if ( v60 != v29 )
  {
    while ( 1 )
    {
      v69 = 6;
      v70 = 0;
      v71 = v29[2];
      if ( v71 != 0 && v71 != -8 && v71 != -16 )
      {
        sub_1649AC0(&v69, *v29 & 0xFFFFFFFFFFFFFFF8LL);
        v72 = v29[3];
        if ( v71 != -8 && v71 != -16 )
        {
          if ( v71 )
            sub_1649B30(&v69);
        }
      }
      v69 = 6;
      v70 = 0;
      v71 = v29[2];
      if ( v71 == 0 || v71 == -8 || v71 == -16 )
      {
        v72 = v29[3];
        v31 = sub_13995B0((__int64)&v69);
      }
      else
      {
        sub_1649AC0(&v69, *v29 & 0xFFFFFFFFFFFFFFF8LL);
        v30 = v71 != -16 && v71 != 0 && v71 != -8;
        v72 = v29[3];
        v31 = sub_13995B0((__int64)&v69);
        if ( v30 )
          sub_1649B30(&v69);
      }
      if ( v31 )
      {
        v69 = (__int64)&v71;
        sub_1399600(&v69, byte_3F871B3, (__int64)byte_3F871B3);
        v32 = v70;
        if ( (__int64 *)v69 != &v71 )
          j_j___libc_free_0(v69, v71 + 1);
        v33 = v32 == 0;
        v34 = -1;
        v69 = (__int64)&v71;
        if ( !v33 )
          v34 = v28;
        sub_1399600(&v69, byte_3F871B3, (__int64)byte_3F871B3);
        v35 = *a1;
        v36 = *(_QWORD *)(*a1 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v36) <= 4 )
        {
          v35 = sub_16E7EE0(v35, "\tNode", 5);
        }
        else
        {
          *(_DWORD *)v36 = 1685016073;
          *(_BYTE *)(v36 + 4) = 101;
          *(_QWORD *)(v35 + 24) += 5LL;
        }
        sub_16E7B40(v35, a2);
        if ( v34 != -1 )
        {
          v41 = *a1;
          v42 = *(_WORD **)(*a1 + 24);
          if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v42 <= 1u )
          {
            v41 = sub_16E7EE0(v41, ":s", 2);
          }
          else
          {
            *v42 = 29498;
            *(_QWORD *)(v41 + 24) += 2LL;
          }
          sub_16E7AB0(v41, v34);
        }
        v37 = *a1;
        v38 = *(_QWORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v38 <= 7u )
        {
          v37 = sub_16E7EE0(v37, " -> Node", 8);
        }
        else
        {
          *v38 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v37 + 24) += 8LL;
        }
        sub_16E7B40(v37, v31);
        if ( v70 )
        {
          v43 = *a1;
          v44 = *(_BYTE **)(*a1 + 24);
          if ( *(_BYTE **)(*a1 + 16) == v44 )
          {
            v43 = sub_16E7EE0(v43, "[", 1);
          }
          else
          {
            *v44 = 91;
            ++*(_QWORD *)(v43 + 24);
          }
          v45 = sub_16E7EE0(v43, (const char *)v69, v70);
          v46 = *(_BYTE **)(v45 + 24);
          if ( *(_BYTE **)(v45 + 16) == v46 )
          {
            sub_16E7EE0(v45, "]", 1);
          }
          else
          {
            *v46 = 93;
            ++*(_QWORD *)(v45 + 24);
          }
        }
        v39 = *a1;
        v40 = *(_WORD **)(*a1 + 24);
        if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v40 <= 1u )
        {
          sub_16E7EE0(v39, ";\n", 2);
        }
        else
        {
          *v40 = 2619;
          *(_QWORD *)(v39 + 24) += 2LL;
        }
        if ( (__int64 *)v69 != &v71 )
          j_j___libc_free_0(v69, v71 + 1);
      }
      ++v28;
      v29 += 4;
      if ( v28 == 64 )
        break;
      if ( v60 == v29 )
        goto LABEL_101;
    }
    if ( v60 != v29 )
    {
      v47 = v29;
      do
      {
        v69 = 6;
        v70 = 0;
        v71 = v47[2];
        if ( v71 != 0 && v71 != -8 && v71 != -16 )
        {
          sub_1649AC0(&v69, *v47 & 0xFFFFFFFFFFFFFFF8LL);
          v72 = v47[3];
          if ( v71 != 0 && v71 != -8 && v71 != -16 )
            sub_1649B30(&v69);
        }
        v48 = v47;
        v47 += 4;
        sub_139A3C0(a1, (__int64)a2, 64, v48, (__int64 (__fastcall *)(__int64 *))sub_13995B0);
      }
      while ( v60 != v47 );
    }
  }
LABEL_101:
  result = sub_16E7BC0(&v73);
  if ( (_QWORD *)v64[0] != v65 )
    result = j_j___libc_free_0(v64[0], v65[0] + 1LL);
  if ( (_QWORD *)v62[0] != v63 )
    return j_j___libc_free_0(v62[0], v63[0] + 1LL);
  return result;
}
