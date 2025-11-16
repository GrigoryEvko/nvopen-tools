// Function: sub_1BCAF90
// Address: 0x1bcaf90
//
void *__fastcall sub_1BCAF90(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rax
  void *v5; // rdx
  __int64 v6; // rdi
  _QWORD *v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 *v10; // rdi
  __int64 *v11; // r14
  __int64 v12; // r15
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rsi
  signed __int64 v17; // rdx
  _QWORD *v18; // rdx
  _BYTE *v19; // r8
  _BYTE *v20; // rdi
  __int64 v21; // rbx
  char v22; // r15
  __int64 v23; // r12
  __int64 v24; // rax
  char **v25; // rdi
  __int64 v26; // r15
  _BYTE *v27; // rax
  _WORD *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rdx
  int v31; // r13d
  int *v32; // rbx
  __int64 v33; // rdi
  _QWORD *v34; // rdx
  __int64 v35; // rdi
  _WORD *v36; // rdx
  __int64 v37; // r12
  size_t v38; // rax
  int v39; // esi
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rdi
  _WORD *v43; // rdx
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 v46; // rdi
  _BYTE *v47; // rax
  void *result; // rax
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // r12
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdi
  _QWORD *v57; // rdx
  __int64 v58; // rdi
  _WORD *v59; // rdx
  size_t v60; // r13
  int v61; // r13d
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rdi
  _WORD *v65; // rdx
  __int64 v66; // rdi
  _BYTE *v67; // rax
  __int64 v68; // rax
  size_t v70; // [rsp+20h] [rbp-110h]
  __int64 v71; // [rsp+28h] [rbp-108h]
  _QWORD *v72; // [rsp+28h] [rbp-108h]
  __int64 v73; // [rsp+38h] [rbp-F8h]
  __int64 *v74; // [rsp+40h] [rbp-F0h]
  __int64 v75; // [rsp+40h] [rbp-F0h]
  int *v76; // [rsp+40h] [rbp-F0h]
  size_t v78; // [rsp+58h] [rbp-D8h]
  char v79[16]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v80[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v81[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v82; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v83; // [rsp+98h] [rbp-98h]
  _QWORD v84[2]; // [rsp+A0h] [rbp-90h] BYREF
  char *v85; // [rsp+B0h] [rbp-80h] BYREF
  size_t v86; // [rsp+B8h] [rbp-78h]
  _QWORD v87[2]; // [rsp+C0h] [rbp-70h] BYREF
  char *v88; // [rsp+D0h] [rbp-60h] BYREF
  _BYTE *v89; // [rsp+D8h] [rbp-58h]
  _BYTE *v90; // [rsp+E0h] [rbp-50h] BYREF
  _BYTE *v91; // [rsp+E8h] [rbp-48h]
  int v92; // [rsp+F0h] [rbp-40h]
  char **v93; // [rsp+F8h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 88) )
  {
    strcpy(v79, "color=red");
    v78 = 9;
  }
  else
  {
    v78 = 0;
    v79[0] = 0;
  }
  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v3) <= 4 )
  {
    v2 = sub_16E7EE0(v2, "\tNode", 5u);
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
    sub_16E7EE0(v4, " [shape=record,", 0xFu);
  }
  else
  {
    qmemcpy(v5, " [shape=record,", 15);
    *(_QWORD *)(v4 + 24) += 15LL;
  }
  if ( v78 )
  {
    v55 = sub_16E7EE0(*(_QWORD *)a1, v79, v78);
    sub_1263B40(v55, ",");
  }
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
  if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v7 <= 7u )
  {
    sub_16E7EE0(v6, "label=\"{", 8u);
  }
  else
  {
    *v7 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v6 + 24) += 8LL;
  }
  v8 = **(_QWORD **)(a1 + 8);
  v71 = *(_QWORD *)a1;
  v85 = (char *)v87;
  v86 = 0;
  v88 = (char *)&unk_49EFBE0;
  LOBYTE(v87[0]) = 0;
  v9 = *(unsigned int *)(a2 + 8);
  v92 = 1;
  v91 = 0;
  v10 = *(__int64 **)a2;
  v90 = 0;
  v89 = 0;
  v93 = &v85;
  if ( (unsigned int)v9 > 1 )
  {
    v11 = v10 + 1;
    v12 = *v10;
    v13 = v10 + 1;
    while ( *v13 == v12 )
    {
      if ( &v10[(unsigned int)(v9 - 2) + 2] == ++v13 )
        goto LABEL_126;
    }
    v74 = &v10[v9];
    while ( 1 )
    {
      sub_155C2B0(v12, (__int64)&v88, 0);
      v14 = *(_QWORD **)(v8 + 384);
      v15 = 3LL * *(unsigned int *)(v8 + 392);
      v16 = &v14[v15];
      v17 = 0xAAAAAAAAAAAAAAABLL * ((v15 * 8) >> 3);
      if ( v17 >> 2 )
      {
        v18 = &v14[12 * (v17 >> 2)];
        while ( 1 )
        {
          if ( v12 == *v14 )
            goto LABEL_23;
          if ( v12 == v14[3] )
          {
            v14 += 3;
LABEL_23:
            v19 = v91;
            goto LABEL_24;
          }
          if ( v12 == v14[6] )
          {
            v19 = v91;
            v14 += 6;
            goto LABEL_24;
          }
          if ( v12 == v14[9] )
            break;
          v14 += 12;
          if ( v18 == v14 )
          {
            v17 = 0xAAAAAAAAAAAAAAABLL * (v16 - v14);
            goto LABEL_60;
          }
        }
        v19 = v91;
        v14 += 9;
      }
      else
      {
LABEL_60:
        v19 = v91;
        v20 = v91;
        if ( v17 == 2 )
          goto LABEL_114;
        if ( v17 == 3 )
        {
          if ( v12 != *v14 )
          {
            v14 += 3;
LABEL_114:
            if ( v12 != *v14 )
            {
              v14 += 3;
              if ( v12 != *v14 )
                goto LABEL_27;
            }
          }
        }
        else if ( v17 != 1 || v12 != *v14 )
        {
          goto LABEL_27;
        }
      }
LABEL_24:
      v20 = v19;
      if ( v16 != v14 )
      {
        if ( (unsigned __int64)(v90 - v19) <= 9 )
        {
          sub_16E7EE0((__int64)&v88, " <extract>", 0xAu);
          v20 = v91;
        }
        else
        {
          qmemcpy(v19, " <extract>", 10);
          v20 = v91 + 10;
          v91 += 10;
        }
      }
LABEL_27:
      if ( v90 == v20 )
      {
        sub_16E7EE0((__int64)&v88, "\n", 1u);
        if ( v74 == v11 )
          goto LABEL_31;
      }
      else
      {
        *v20 = 10;
        ++v91;
        if ( v74 == v11 )
          goto LABEL_31;
      }
      v12 = *v11++;
    }
  }
LABEL_126:
  v54 = sub_1263B40((__int64)&v88, "<splat> ");
  sub_155C2B0(**(_QWORD **)a2, v54, 0);
LABEL_31:
  sub_16E7BC0((__int64 *)&v88);
  sub_16BE9B0((__int64 *)&v88, (__int64)&v85);
  sub_16E7EE0(v71, v88, (size_t)v89);
  if ( v88 != (char *)&v90 )
    j_j___libc_free_0(v88, v90 + 1);
  if ( v85 != (char *)v87 )
    j_j___libc_free_0(v85, v87[0] + 1LL);
  v82 = v84;
  sub_1BB98B0((__int64 *)&v82, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v83 )
  {
    v53 = sub_1263B40(*(_QWORD *)a1, "|");
    sub_16BE9B0((__int64 *)&v88, (__int64)&v82);
    sub_16E7EE0(v53, v88, (size_t)v89);
    if ( v88 != (char *)&v90 )
      j_j___libc_free_0(v88, v90 + 1);
  }
  v85 = (char *)v87;
  sub_1BB98B0((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v86 )
  {
    v52 = sub_1263B40(*(_QWORD *)a1, "|");
    sub_16BE9B0((__int64 *)&v88, (__int64)&v85);
    sub_16E7EE0(v52, v88, (size_t)v89);
    if ( v88 != (char *)&v90 )
      j_j___libc_free_0(v88, v90 + 1);
  }
  if ( v85 != (char *)v87 )
    j_j___libc_free_0(v85, v87[0] + 1LL);
  if ( v82 != v84 )
    j_j___libc_free_0(v82, v84[0] + 1LL);
  v80[1] = 0;
  v21 = 0;
  v93 = (char **)v80;
  v22 = 0;
  v23 = *(_QWORD *)(a2 + 152);
  v80[0] = v81;
  v24 = *(unsigned int *)(a2 + 160);
  LOBYTE(v81[0]) = 0;
  v92 = 1;
  v91 = 0;
  v90 = 0;
  v89 = 0;
  v88 = (char *)&unk_49EFBE0;
  v75 = v23 + 4 * v24;
  if ( v23 == v75 )
    goto LABEL_72;
  do
  {
    v82 = v84;
    sub_1BB98B0((__int64 *)&v82, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v83 )
    {
      v28 = v91;
      if ( v21 )
      {
        if ( v90 != v91 )
        {
          *v91 = 124;
          v28 = v91 + 1;
          v91 = v28;
          if ( (unsigned __int64)(v90 - (_BYTE *)v28) <= 1 )
            goto LABEL_58;
          goto LABEL_44;
        }
        sub_16E7EE0((__int64)&v88, "|", 1u);
        v28 = v91;
      }
      if ( (unsigned __int64)(v90 - (_BYTE *)v28) <= 1 )
      {
LABEL_58:
        v25 = (char **)sub_16E7EE0((__int64)&v88, "<s", 2u);
        goto LABEL_45;
      }
LABEL_44:
      v25 = &v88;
      *v28 = 29500;
      v91 += 2;
LABEL_45:
      v26 = sub_16E7A90((__int64)v25, v21);
      v27 = *(_BYTE **)(v26 + 24);
      if ( *(_BYTE **)(v26 + 16) == v27 )
      {
        v26 = sub_16E7EE0(v26, ">", 1u);
      }
      else
      {
        *v27 = 62;
        ++*(_QWORD *)(v26 + 24);
      }
      sub_16BE9B0((__int64 *)&v85, (__int64)&v82);
      sub_16E7EE0(v26, v85, v86);
      if ( v85 != (char *)v87 )
        j_j___libc_free_0(v85, v87[0] + 1LL);
      if ( v82 != v84 )
        j_j___libc_free_0(v82, v84[0] + 1LL);
      v22 = 1;
LABEL_52:
      v23 += 4;
      if ( v75 == v23 )
        goto LABEL_71;
      goto LABEL_53;
    }
    if ( v82 == v84 )
      goto LABEL_52;
    v23 += 4;
    j_j___libc_free_0(v82, v84[0] + 1LL);
    if ( v75 == v23 )
    {
LABEL_71:
      if ( !v22 )
        goto LABEL_72;
      goto LABEL_109;
    }
LABEL_53:
    ++v21;
  }
  while ( v21 != 64 );
  if ( !v22 )
    goto LABEL_72;
  sub_1263B40((__int64)&v88, "|<s64>truncated...");
LABEL_109:
  sub_1263B40(*(_QWORD *)a1, "|");
  v49 = sub_1263B40(*(_QWORD *)a1, "{");
  if ( v91 != v89 )
    sub_16E7BA0((__int64 *)&v88);
  v50 = sub_16E7EE0(v49, *v93, (size_t)v93[1]);
  sub_1263B40(v50, "}");
LABEL_72:
  v29 = *(_QWORD *)a1;
  v30 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v30) <= 4 )
  {
    sub_16E7EE0(v29, "}\"];\n", 5u);
  }
  else
  {
    *(_DWORD *)v30 = 995959421;
    *(_BYTE *)(v30 + 4) = 10;
    *(_QWORD *)(v29 + 24) += 5LL;
  }
  v31 = 0;
  v32 = *(int **)(a2 + 152);
  v76 = &v32[*(unsigned int *)(a2 + 160)];
  if ( v76 != v32 )
  {
    v72 = *(_QWORD **)(a2 + 144);
    while ( 1 )
    {
      v37 = *v72 + 176LL * *v32;
      if ( v37 )
      {
        v85 = (char *)v87;
        sub_1BB98B0((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
        v38 = v86;
        if ( v85 != (char *)v87 )
        {
          v70 = v86;
          j_j___libc_free_0(v85, v87[0] + 1LL);
          v38 = v70;
        }
        v39 = -1;
        v85 = (char *)v87;
        if ( v38 )
          v39 = v31;
        sub_1BB98B0((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
        v40 = *(_QWORD *)a1;
        v41 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v41) > 4 )
        {
          *(_DWORD *)v41 = 1685016073;
          *(_BYTE *)(v41 + 4) = 101;
          *(_QWORD *)(v40 + 24) += 5LL;
        }
        else
        {
          v40 = sub_16E7EE0(v40, "\tNode", 5u);
        }
        sub_16E7B40(v40, a2);
        if ( v39 != -1 )
        {
          v42 = *(_QWORD *)a1;
          v43 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v43 <= 1u )
          {
            v51 = sub_16E7EE0(v42, ":s", 2u);
            sub_16E7AB0(v51, v39);
          }
          else
          {
            *v43 = 29498;
            *(_QWORD *)(v42 + 24) += 2LL;
            sub_16E7AB0(v42, v39);
          }
        }
        v33 = *(_QWORD *)a1;
        v34 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v34 <= 7u )
        {
          v33 = sub_16E7EE0(v33, " -> Node", 8u);
        }
        else
        {
          *v34 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v33 + 24) += 8LL;
        }
        sub_16E7B40(v33, v37);
        if ( v86 )
        {
          v44 = *(_QWORD *)a1;
          v45 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v45 )
          {
            v44 = sub_16E7EE0(v44, "[", 1u);
          }
          else
          {
            *v45 = 91;
            ++*(_QWORD *)(v44 + 24);
          }
          v46 = sub_16E7EE0(v44, v85, v86);
          v47 = *(_BYTE **)(v46 + 24);
          if ( *(_BYTE **)(v46 + 16) == v47 )
          {
            sub_16E7EE0(v46, "]", 1u);
          }
          else
          {
            *v47 = 93;
            ++*(_QWORD *)(v46 + 24);
          }
        }
        v35 = *(_QWORD *)a1;
        v36 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v36 <= 1u )
        {
          sub_16E7EE0(v35, ";\n", 2u);
        }
        else
        {
          *v36 = 2619;
          *(_QWORD *)(v35 + 24) += 2LL;
        }
        if ( v85 != (char *)v87 )
          j_j___libc_free_0(v85, v87[0] + 1LL);
      }
      ++v32;
      ++v31;
      if ( v76 == v32 )
        break;
      if ( v31 == 64 )
      {
        do
        {
          v73 = *v72 + 176LL * *v32;
          if ( v73 )
          {
            v85 = (char *)v87;
            sub_1BB98B0((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
            v60 = v86;
            if ( v85 != (char *)v87 )
              j_j___libc_free_0(v85, v87[0] + 1LL);
            v85 = (char *)v87;
            sub_1BB98B0((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
            v61 = v60 == 0 ? -1 : 0x40;
            v62 = *(_QWORD *)a1;
            v63 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) - v63) > 4 )
            {
              *(_DWORD *)v63 = 1685016073;
              *(_BYTE *)(v63 + 4) = 101;
              *(_QWORD *)(v62 + 24) += 5LL;
            }
            else
            {
              v62 = sub_16E7EE0(v62, "\tNode", 5u);
            }
            sub_16E7B40(v62, a2);
            if ( v61 != -1 )
            {
              v64 = *(_QWORD *)a1;
              v65 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
              if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v65 <= 1u )
              {
                v64 = sub_16E7EE0(v64, ":s", 2u);
              }
              else
              {
                *v65 = 29498;
                *(_QWORD *)(v64 + 24) += 2LL;
              }
              sub_16E7AB0(v64, 64);
            }
            v56 = *(_QWORD *)a1;
            v57 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v57 <= 7u )
            {
              v56 = sub_16E7EE0(v56, " -> Node", 8u);
            }
            else
            {
              *v57 = 0x65646F4E203E2D20LL;
              *(_QWORD *)(v56 + 24) += 8LL;
            }
            sub_16E7B40(v56, v73);
            if ( v86 )
            {
              v66 = *(_QWORD *)a1;
              v67 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
              if ( *(_BYTE **)(*(_QWORD *)a1 + 16LL) == v67 )
              {
                v66 = sub_16E7EE0(v66, "[", 1u);
              }
              else
              {
                *v67 = 91;
                ++*(_QWORD *)(v66 + 24);
              }
              v68 = sub_16E7EE0(v66, v85, v86);
              sub_1263B40(v68, "]");
            }
            v58 = *(_QWORD *)a1;
            v59 = *(_WORD **)(*(_QWORD *)a1 + 24LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 16LL) - (_QWORD)v59 <= 1u )
            {
              sub_16E7EE0(v58, ";\n", 2u);
            }
            else
            {
              *v59 = 2619;
              *(_QWORD *)(v58 + 24) += 2LL;
            }
            if ( v85 != (char *)v87 )
              j_j___libc_free_0(v85, v87[0] + 1LL);
          }
          ++v32;
        }
        while ( v76 != v32 );
        break;
      }
    }
  }
  result = sub_16E7BC0((__int64 *)&v88);
  if ( (_QWORD *)v80[0] != v81 )
    return (void *)j_j___libc_free_0(v80[0], v81[0] + 1LL);
  return result;
}
