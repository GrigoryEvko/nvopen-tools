// Function: sub_137FA70
// Address: 0x137fa70
//
__int64 __fastcall sub_137FA70(_BYTE *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  void *v6; // rdx
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rbx
  char v15; // r13
  const char **v16; // rdi
  __int64 v17; // r13
  _BYTE *v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdx
  unsigned __int64 v21; // r10
  unsigned int v22; // r15d
  __int64 v23; // rdx
  unsigned int v24; // r13d
  unsigned __int64 v25; // r14
  unsigned int v26; // ebx
  unsigned int v27; // eax
  char v28; // al
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r13
  unsigned int v34; // ebx
  __int64 v35; // rdi
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  _WORD *v38; // rdx
  __int64 v39; // r12
  __int64 v40; // rax
  int v41; // esi
  __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rdi
  _WORD *v45; // rdx
  __int64 result; // rax
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __int64 v49; // rdi
  _BYTE *v50; // rax
  __m128i *v51; // rdx
  __m128i si128; // xmm0
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdi
  _QWORD *v56; // rdx
  __int64 v57; // rdi
  _WORD *v58; // rdx
  __int64 v59; // r12
  __int64 v60; // r14
  int v61; // r14d
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdi
  _WORD *v66; // rdx
  __int64 v67; // rdi
  _BYTE *v68; // rax
  __int64 v69; // rdi
  _BYTE *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdi
  _BYTE *v73; // rax
  __int64 v74; // r12
  _BYTE *v75; // rax
  __int64 v76; // rdi
  _BYTE *v77; // rax
  int v79; // [rsp+18h] [rbp-108h]
  __int64 v80; // [rsp+18h] [rbp-108h]
  __int64 v81; // [rsp+28h] [rbp-F8h]
  __int64 v82; // [rsp+28h] [rbp-F8h]
  int v83; // [rsp+30h] [rbp-F0h]
  _QWORD v85[2]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD v86[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v87; // [rsp+80h] [rbp-A0h] BYREF
  unsigned __int64 v88; // [rsp+88h] [rbp-98h]
  _QWORD v89[2]; // [rsp+90h] [rbp-90h] BYREF
  const char *v90; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-78h]
  _QWORD v92[2]; // [rsp+B0h] [rbp-70h] BYREF
  const char *v93; // [rsp+C0h] [rbp-60h] BYREF
  _BYTE *v94; // [rsp+C8h] [rbp-58h]
  _BYTE *v95; // [rsp+D0h] [rbp-50h] BYREF
  _BYTE *v96; // [rsp+D8h] [rbp-48h]
  int v97; // [rsp+E0h] [rbp-40h]
  const char **v98; // [rsp+E8h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)a1;
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
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v8 <= 7u )
  {
    sub_16E7EE0(v7, "label=\"{", 8);
  }
  else
  {
    *v8 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v7 + 24) += 8LL;
  }
  v81 = *(_QWORD *)a1;
  if ( !a1[16] )
  {
    LOBYTE(v92[0]) = 0;
    v90 = (const char *)v92;
    v91 = 0;
    v97 = 1;
    v96 = 0;
    v95 = 0;
    v94 = 0;
    v93 = (const char *)&unk_49EFBE0;
    v98 = &v90;
    sub_1649960(a2);
    if ( !v20 )
    {
      sub_15537D0(a2, &v93, 0);
      if ( v95 == v96 )
        sub_16E7EE0(&v93, ":", 1);
      else
        *v96++ = 58;
    }
    sub_155C2B0(a2, &v93, 0);
    if ( v96 != v94 )
      sub_16E7BA0(&v93);
    v87 = v89;
    sub_137EA90((__int64 *)&v87, *v98, (__int64)&v98[1][(_QWORD)*v98]);
    if ( *(_BYTE *)v87 == 10 )
    {
      sub_2240CE0(&v87, 0, 1);
      v21 = v88;
      if ( v88 )
      {
LABEL_40:
        v22 = 0;
        v23 = (__int64)v87;
        v24 = 0;
        v25 = 0;
        v26 = 0;
        do
        {
          v28 = *(_BYTE *)(v23 + v25);
          if ( v28 == 10 )
          {
            *(_BYTE *)(v23 + v25) = 92;
            v22 = 0;
            v26 = 0;
            sub_2240FD0(&v87, v25 + 1, 0, 1, 108);
            v27 = v24++;
            v23 = (__int64)v87;
            v21 = v88;
          }
          else if ( v28 == 59 )
          {
            v53 = (unsigned int)sub_22417D0(&v87, 10, v24 + 1);
            if ( v53 == v88 )
            {
              v88 = v25;
              *((_BYTE *)v87 + v25) = 0;
            }
            else
            {
              sub_2240CE0(&v87, v25, v53 - v25);
            }
            v23 = (__int64)v87;
            v21 = v88;
            v27 = v24 - 1;
          }
          else if ( v26 == 80 )
          {
            if ( !v22 )
              v22 = v24;
            v29 = v22;
            if ( v21 < v22 )
              sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
            v26 = v24 - v22;
            v22 = 0;
            sub_2241130(&v87, v29, 0, "\\l...", 5);
            v27 = v24 + 3;
            v24 += 4;
            v23 = (__int64)v87;
            v21 = v88;
          }
          else
          {
            v27 = v24;
            ++v26;
            ++v24;
          }
          v25 = v24;
          if ( *(_BYTE *)(v23 + v27) == 32 )
            v22 = v27;
        }
        while ( v24 != v21 );
        v2 = a2;
      }
    }
    else
    {
      v21 = v88;
      if ( v88 )
        goto LABEL_40;
    }
LABEL_53:
    sub_16E7BC0(&v93);
    if ( v90 != (const char *)v92 )
      j_j___libc_free_0(v90, v92[0] + 1LL);
    goto LABEL_11;
  }
  sub_1649960(a2);
  if ( !v9 )
  {
    LOBYTE(v92[0]) = 0;
    v98 = &v90;
    v90 = (const char *)v92;
    v93 = (const char *)&unk_49EFBE0;
    v91 = 0;
    v97 = 1;
    v96 = 0;
    v95 = 0;
    v94 = 0;
    sub_15537D0(a2, &v93, 0);
    if ( v96 != v94 )
      sub_16E7BA0(&v93);
    v87 = v89;
    sub_137EA90((__int64 *)&v87, *v98, (__int64)&v98[1][(_QWORD)*v98]);
    goto LABEL_53;
  }
  v11 = (_BYTE *)sub_1649960(a2);
  if ( !v11 )
  {
    LOBYTE(v89[0]) = 0;
    v87 = v89;
    v88 = 0;
    sub_16BE9B0(&v93, &v87);
    goto LABEL_12;
  }
  v87 = v89;
  sub_137E9E0((__int64 *)&v87, v11, (__int64)&v11[v10]);
LABEL_11:
  sub_16BE9B0(&v93, &v87);
LABEL_12:
  sub_16E7EE0(v81, v93, v94);
  if ( v93 != (const char *)&v95 )
    j_j___libc_free_0(v93, v95 + 1);
  if ( v87 != v89 )
    j_j___libc_free_0(v87, v89[0] + 1LL);
  v85[1] = 0;
  v85[0] = v86;
  LOBYTE(v86[0]) = 0;
  v97 = 1;
  v96 = 0;
  v95 = 0;
  v94 = 0;
  v93 = (const char *)&unk_49EFBE0;
  v98 = (const char **)v85;
  v12 = sub_157EBA0(v2);
  if ( !v12 )
    goto LABEL_59;
  v13 = sub_15F4D60(v12);
  v79 = v13;
  if ( !v13 )
    goto LABEL_59;
  v14 = 0;
  v15 = 0;
  v82 = (unsigned int)(v13 - 1);
  do
  {
    sub_137EB40((__int64)&v87, v2, v14);
    if ( v88 )
    {
      v19 = v96;
      if ( (_DWORD)v14 )
      {
        if ( v95 != v96 )
        {
          *v96 = 124;
          v19 = v96 + 1;
          v96 = v19;
          if ( (unsigned __int64)(v95 - (_BYTE *)v19) <= 1 )
            goto LABEL_34;
          goto LABEL_20;
        }
        sub_16E7EE0(&v93, "|", 1);
        v19 = v96;
      }
      if ( (unsigned __int64)(v95 - (_BYTE *)v19) <= 1 )
      {
LABEL_34:
        v16 = (const char **)sub_16E7EE0(&v93, "<s", 2);
        goto LABEL_21;
      }
LABEL_20:
      v16 = &v93;
      *v19 = 29500;
      v96 += 2;
LABEL_21:
      v17 = sub_16E7A90(v16, v14);
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
      sub_16BE9B0(&v90, &v87);
      sub_16E7EE0(v17, v90, v91);
      if ( v90 != (const char *)v92 )
        j_j___libc_free_0(v90, v92[0] + 1LL);
      if ( v87 != v89 )
        j_j___libc_free_0(v87, v89[0] + 1LL);
      v15 = 1;
LABEL_28:
      if ( v82 == v14 )
        goto LABEL_58;
      goto LABEL_29;
    }
    if ( v87 == v89 )
      goto LABEL_28;
    j_j___libc_free_0(v87, v89[0] + 1LL);
    if ( v82 == v14 )
      goto LABEL_58;
LABEL_29:
    ++v14;
  }
  while ( v14 != 64 );
  if ( v79 == 64 )
  {
LABEL_58:
    if ( !v15 )
      goto LABEL_59;
    goto LABEL_146;
  }
  if ( !v15 )
    goto LABEL_59;
  v51 = (__m128i *)v96;
  if ( (unsigned __int64)(v95 - v96) <= 0x11 )
  {
    sub_16E7EE0(&v93, "|<s64>truncated...", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
    *((_WORD *)v96 + 8) = 11822;
    *v51 = si128;
    v96 += 18;
  }
LABEL_146:
  v72 = *(_QWORD *)a1;
  v73 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v73 )
  {
    sub_16E7EE0(v72, "|", 1);
  }
  else
  {
    *v73 = 124;
    ++*(_QWORD *)(v72 + 24);
  }
  v74 = *(_QWORD *)a1;
  v75 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v75 )
  {
    v74 = sub_16E7EE0(*(_QWORD *)a1, "{", 1);
  }
  else
  {
    *v75 = 123;
    ++*(_QWORD *)(v74 + 24);
  }
  if ( v96 != v94 )
    sub_16E7BA0(&v93);
  v76 = sub_16E7EE0(v74, *v98, v98[1]);
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
LABEL_59:
  v30 = *(_QWORD *)a1;
  v31 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v31) <= 4 )
  {
    sub_16E7EE0(v30, "}\"];\n", 5);
  }
  else
  {
    *(_DWORD *)v31 = 995959421;
    *(_BYTE *)(v31 + 4) = 10;
    *(_QWORD *)(v30 + 24) += 5LL;
  }
  v32 = sub_157EBA0(v2);
  v33 = v32;
  if ( v32 )
  {
    v83 = sub_15F4D60(v32);
    if ( v83 )
    {
      v34 = 0;
      do
      {
        sub_15F4DF0(v33, v34);
        v39 = sub_15F4DF0(v33, v34);
        if ( v39 )
        {
          sub_137EB40((__int64)&v90, v2, v34);
          v40 = v91;
          if ( v90 != (const char *)v92 )
          {
            v80 = v91;
            j_j___libc_free_0(v90, v92[0] + 1LL);
            v40 = v80;
          }
          v41 = -1;
          if ( v40 )
            v41 = v34;
          sub_137ED00((__int64)&v90, v2, v34);
          v42 = *(_QWORD *)a1;
          v43 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v43) > 4 )
          {
            *(_DWORD *)v43 = 1685016073;
            *(_BYTE *)(v43 + 4) = 101;
            *(_QWORD *)(v42 + 24) += 5LL;
          }
          else
          {
            v42 = sub_16E7EE0(v42, "\tNode", 5);
          }
          sub_16E7B40(v42, v2);
          if ( v41 != -1 )
          {
            v44 = *(_QWORD *)a1;
            v45 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v45 <= 1u )
            {
              v54 = sub_16E7EE0(v44, ":s", 2);
              sub_16E7AB0(v54, v41);
            }
            else
            {
              *v45 = 29498;
              *(_QWORD *)(v44 + 24) += 2LL;
              sub_16E7AB0(v44, v41);
            }
          }
          v35 = *(_QWORD *)a1;
          v36 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v36 <= 7u )
          {
            v35 = sub_16E7EE0(v35, " -> Node", 8);
          }
          else
          {
            *v36 = 0x65646F4E203E2D20LL;
            *(_QWORD *)(v35 + 24) += 8LL;
          }
          sub_16E7B40(v35, v39);
          if ( v91 )
          {
            v47 = *(_QWORD *)a1;
            v48 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v48 )
            {
              v47 = sub_16E7EE0(v47, "[", 1);
            }
            else
            {
              *v48 = 91;
              ++*(_QWORD *)(v47 + 24);
            }
            v49 = sub_16E7EE0(v47, v90, v91);
            v50 = *(_BYTE **)(v49 + 24);
            if ( *(_BYTE **)(v49 + 16) == v50 )
            {
              sub_16E7EE0(v49, "]", 1);
            }
            else
            {
              *v50 = 93;
              ++*(_QWORD *)(v49 + 24);
            }
          }
          v37 = *(_QWORD *)a1;
          v38 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v38 <= 1u )
          {
            sub_16E7EE0(v37, ";\n", 2);
          }
          else
          {
            *v38 = 2619;
            *(_QWORD *)(v37 + 24) += 2LL;
          }
          if ( v90 != (const char *)v92 )
            j_j___libc_free_0(v90, v92[0] + 1LL);
        }
        if ( ++v34 == v83 )
          goto LABEL_85;
      }
      while ( v34 != 64 );
      if ( v83 != 64 )
      {
        while ( 1 )
        {
          sub_15F4DF0(v33, v34);
          v59 = sub_15F4DF0(v33, v34);
          if ( v59 )
            break;
LABEL_123:
          if ( v83 == ++v34 )
            goto LABEL_85;
        }
        sub_137EB40((__int64)&v90, v2, v34);
        v60 = v91;
        if ( v90 != (const char *)v92 )
          j_j___libc_free_0(v90, v92[0] + 1LL);
        sub_137ED00((__int64)&v90, v2, v34);
        v61 = v60 == 0 ? -1 : 0x40;
        v62 = *(_QWORD *)a1;
        v63 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v63) > 4 )
        {
          *(_DWORD *)v63 = 1685016073;
          *(_BYTE *)(v63 + 4) = 101;
          *(_QWORD *)(v62 + 24) += 5LL;
          sub_16E7B40(v62, v2);
          if ( v61 == -1 )
            goto LABEL_116;
        }
        else
        {
          v64 = sub_16E7EE0(v62, "\tNode", 5);
          sub_16E7B40(v64, v2);
          if ( v61 == -1 )
            goto LABEL_116;
        }
        v65 = *(_QWORD *)a1;
        v66 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v66 <= 1u )
        {
          v71 = sub_16E7EE0(v65, ":s", 2);
          sub_16E7AB0(v71, 64);
        }
        else
        {
          *v66 = 29498;
          *(_QWORD *)(v65 + 24) += 2LL;
          sub_16E7AB0(v65, 64);
        }
LABEL_116:
        v55 = *(_QWORD *)a1;
        v56 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v56 <= 7u )
        {
          v55 = sub_16E7EE0(v55, " -> Node", 8);
        }
        else
        {
          *v56 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v55 + 24) += 8LL;
        }
        sub_16E7B40(v55, v59);
        if ( v91 )
        {
          v67 = *(_QWORD *)a1;
          v68 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v68 )
          {
            v67 = sub_16E7EE0(v67, "[", 1);
          }
          else
          {
            *v68 = 91;
            ++*(_QWORD *)(v67 + 24);
          }
          v69 = sub_16E7EE0(v67, v90, v91);
          v70 = *(_BYTE **)(v69 + 24);
          if ( *(_BYTE **)(v69 + 16) == v70 )
          {
            sub_16E7EE0(v69, "]", 1);
          }
          else
          {
            *v70 = 93;
            ++*(_QWORD *)(v69 + 24);
          }
        }
        v57 = *(_QWORD *)a1;
        v58 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v58 <= 1u )
        {
          sub_16E7EE0(v57, ";\n", 2);
        }
        else
        {
          *v58 = 2619;
          *(_QWORD *)(v57 + 24) += 2LL;
        }
        if ( v90 != (const char *)v92 )
          j_j___libc_free_0(v90, v92[0] + 1LL);
        goto LABEL_123;
      }
    }
  }
LABEL_85:
  result = sub_16E7BC0(&v93);
  if ( (_QWORD *)v85[0] != v86 )
    return j_j___libc_free_0(v85[0], v86[0] + 1LL);
  return result;
}
