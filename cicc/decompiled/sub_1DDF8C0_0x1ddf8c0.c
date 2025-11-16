// Function: sub_1DDF8C0
// Address: 0x1ddf8c0
//
void *__fastcall sub_1DDF8C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 *v4; // r13
  __int64 v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  void *v10; // rdx
  __int64 v11; // rdi
  _QWORD *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  char **v17; // rdi
  __int64 v18; // r14
  _BYTE *v19; // rax
  char *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 *v23; // r12
  __int64 *v24; // rbx
  int v25; // r15d
  void *result; // rax
  __m128i *v27; // rdx
  __m128i si128; // xmm0
  __int64 v29; // rdi
  _BYTE *v30; // rax
  __int64 v31; // r12
  _BYTE *v32; // rax
  __int64 v33; // rdi
  _BYTE *v34; // rax
  unsigned __int64 v35; // rbx
  int v36; // eax
  __int64 v37; // rdi
  _BYTE *v38; // rax
  __int64 v39; // r8
  _BYTE *v40; // rax
  __int64 v41; // r8
  _BYTE *v42; // rax
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // r12
  unsigned __int64 v47; // rax
  __int64 v48; // [rsp+10h] [rbp-120h]
  char v49; // [rsp+37h] [rbp-F9h]
  unsigned int v50; // [rsp+48h] [rbp-E8h]
  __int64 v51; // [rsp+48h] [rbp-E8h]
  __int64 v52; // [rsp+48h] [rbp-E8h]
  __int64 v53; // [rsp+48h] [rbp-E8h]
  char *v54; // [rsp+50h] [rbp-E0h] BYREF
  size_t v55; // [rsp+58h] [rbp-D8h]
  _QWORD v56[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v57[2]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v58[2]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v59; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+98h] [rbp-98h]
  _QWORD v61[2]; // [rsp+A0h] [rbp-90h] BYREF
  char *v62; // [rsp+B0h] [rbp-80h] BYREF
  size_t v63; // [rsp+B8h] [rbp-78h]
  _QWORD v64[2]; // [rsp+C0h] [rbp-70h] BYREF
  char *v65; // [rsp+D0h] [rbp-60h] BYREF
  char *v66; // [rsp+D8h] [rbp-58h]
  char *v67; // [rsp+E0h] [rbp-50h] BYREF
  char *v68; // [rsp+E8h] [rbp-48h]
  int v69; // [rsp+F0h] [rbp-40h]
  char **v70; // [rsp+F8h] [rbp-38h]

  v2 = a1 + 16;
  v3 = a2;
  v4 = (__int64 *)a1;
  v5 = **(_QWORD **)(a1 + 8);
  v54 = (char *)v56;
  v55 = 0;
  v6 = qword_4F982C0[20];
  LOBYTE(v56[0]) = 0;
  v50 = v6;
  if ( v6 )
  {
    if ( !*(_QWORD *)(a1 + 24) )
    {
      v43 = *(_QWORD *)(sub_1DDC510(v5) + 328);
      v44 = sub_1DDC510(v5) + 320;
      if ( v43 != v44 )
      {
        v45 = v43;
        v46 = v44;
        do
        {
          v47 = sub_1DDC3C0(v5, v45);
          if ( v47 < *(_QWORD *)(a1 + 24) )
            v47 = *(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 24) = v47;
          v45 = *(_QWORD *)(v45 + 8);
        }
        while ( v46 != v45 );
        v4 = (__int64 *)a1;
        v3 = a2;
      }
    }
    v35 = sub_1DDC3C0(v5, v3);
    v36 = sub_16AF730(v50, 0x64u);
    v65 = (char *)v4[3];
    if ( sub_16AF500((__int64 *)&v65, v36) <= v35 )
    {
      v69 = 1;
      v68 = 0;
      v65 = (char *)&unk_49EFBE0;
      v67 = 0;
      v66 = 0;
      v70 = &v54;
      sub_16E7EE0((__int64)&v65, "color=\"red\"", 0xBu);
      if ( v68 != v66 )
        sub_16E7BA0((__int64 *)&v65);
      sub_16E7BC0((__int64 *)&v65);
    }
  }
  v7 = *v4;
  v8 = *(_QWORD *)(*v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*v4 + 16) - v8) <= 4 )
  {
    v7 = sub_16E7EE0(v7, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v8 = 1685016073;
    *(_BYTE *)(v8 + 4) = 101;
    *(_QWORD *)(v7 + 24) += 5LL;
  }
  v9 = sub_16E7B40(v7, v3);
  v10 = *(void **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0xEu )
  {
    sub_16E7EE0(v9, " [shape=record,", 0xFu);
  }
  else
  {
    qmemcpy(v10, " [shape=record,", 15);
    *(_QWORD *)(v9 + 24) += 15LL;
  }
  if ( v55 )
  {
    v37 = sub_16E7EE0(*v4, v54, v55);
    v38 = *(_BYTE **)(v37 + 24);
    if ( *(_BYTE **)(v37 + 16) == v38 )
    {
      sub_16E7EE0(v37, ",", 1u);
    }
    else
    {
      *v38 = 44;
      ++*(_QWORD *)(v37 + 24);
    }
  }
  v11 = *v4;
  v12 = *(_QWORD **)(*v4 + 24);
  if ( *(_QWORD *)(*v4 + 16) - (_QWORD)v12 <= 7u )
  {
    sub_16E7EE0(v11, "label=\"{", 8u);
  }
  else
  {
    *v12 = 0x7B223D6C6562616CLL;
    *(_QWORD *)(v11 + 24) += 8LL;
  }
  v13 = *v4;
  sub_1DDD700((__int64)&v62, v2, v3, *(_QWORD *)v4[1]);
  sub_16BE9B0((__int64 *)&v65, (__int64)&v62);
  sub_16E7EE0(v13, v65, (size_t)v66);
  if ( v65 != (char *)&v67 )
    j_j___libc_free_0(v65, v67 + 1);
  if ( v62 != (char *)v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  v59 = v61;
  sub_1DDBCE0((__int64 *)&v59, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v60 )
  {
    v41 = *v4;
    v42 = *(_BYTE **)(*v4 + 24);
    if ( *(_BYTE **)(*v4 + 16) == v42 )
    {
      v41 = sub_16E7EE0(*v4, "|", 1u);
    }
    else
    {
      *v42 = 124;
      ++*(_QWORD *)(v41 + 24);
    }
    v53 = v41;
    sub_16BE9B0((__int64 *)&v65, (__int64)&v59);
    sub_16E7EE0(v53, v65, (size_t)v66);
    if ( v65 != (char *)&v67 )
      j_j___libc_free_0(v65, v67 + 1);
  }
  v62 = (char *)v64;
  sub_1DDBCE0((__int64 *)&v62, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v63 )
  {
    v39 = *v4;
    v40 = *(_BYTE **)(*v4 + 24);
    if ( *(_BYTE **)(*v4 + 16) == v40 )
    {
      v39 = sub_16E7EE0(*v4, "|", 1u);
    }
    else
    {
      *v40 = 124;
      ++*(_QWORD *)(v39 + 24);
    }
    v52 = v39;
    sub_16BE9B0((__int64 *)&v65, (__int64)&v62);
    sub_16E7EE0(v52, v65, (size_t)v66);
    if ( v65 != (char *)&v67 )
      j_j___libc_free_0(v65, v67 + 1);
  }
  if ( v62 != (char *)v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  if ( v59 != v61 )
    j_j___libc_free_0(v59, v61[0] + 1LL);
  v14 = 0;
  v15 = *(_QWORD *)(v3 + 88);
  v70 = (char **)v57;
  v16 = *(_QWORD *)(v3 + 96);
  v57[0] = v58;
  v57[1] = 0;
  LOBYTE(v58[0]) = 0;
  v69 = 1;
  v68 = 0;
  v67 = 0;
  v66 = 0;
  v65 = (char *)&unk_49EFBE0;
  v51 = v16;
  v49 = 0;
  if ( v15 == v16 )
    goto LABEL_41;
  v48 = v3;
  do
  {
    v59 = v61;
    sub_1DDBCE0((__int64 *)&v59, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v60 )
    {
      v20 = v68;
      if ( v14 )
      {
        if ( v67 != v68 )
        {
          *v68 = 124;
          v20 = v68 + 1;
          v68 = v20;
          if ( (unsigned __int64)(v67 - v20) <= 1 )
            goto LABEL_36;
          goto LABEL_22;
        }
        sub_16E7EE0((__int64)&v65, "|", 1u);
        v20 = v68;
      }
      if ( (unsigned __int64)(v67 - v20) <= 1 )
      {
LABEL_36:
        v17 = (char **)sub_16E7EE0((__int64)&v65, "<s", 2u);
        goto LABEL_23;
      }
LABEL_22:
      v17 = &v65;
      *(_WORD *)v20 = 29500;
      v68 += 2;
LABEL_23:
      v18 = sub_16E7A90((__int64)v17, v14);
      v19 = *(_BYTE **)(v18 + 24);
      if ( *(_BYTE **)(v18 + 16) == v19 )
      {
        v18 = sub_16E7EE0(v18, ">", 1u);
      }
      else
      {
        *v19 = 62;
        ++*(_QWORD *)(v18 + 24);
      }
      sub_16BE9B0((__int64 *)&v62, (__int64)&v59);
      sub_16E7EE0(v18, v62, v63);
      if ( v62 != (char *)v64 )
        j_j___libc_free_0(v62, v64[0] + 1LL);
      if ( v59 != v61 )
        j_j___libc_free_0(v59, v61[0] + 1LL);
      v49 = 1;
LABEL_30:
      v15 += 8;
      if ( v51 == v15 )
        goto LABEL_40;
      goto LABEL_31;
    }
    if ( v59 == v61 )
      goto LABEL_30;
    v15 += 8;
    j_j___libc_free_0(v59, v61[0] + 1LL);
    if ( v51 == v15 )
    {
LABEL_40:
      v3 = v48;
      if ( !v49 )
        goto LABEL_41;
      goto LABEL_58;
    }
LABEL_31:
    ++v14;
  }
  while ( v14 != 64 );
  v3 = v48;
  if ( !v49 )
    goto LABEL_41;
  v27 = (__m128i *)v68;
  if ( (unsigned __int64)(v67 - v68) <= 0x11 )
  {
    sub_16E7EE0((__int64)&v65, "|<s64>truncated...", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
    *((_WORD *)v68 + 8) = 11822;
    *v27 = si128;
    v68 += 18;
  }
LABEL_58:
  v29 = *v4;
  v30 = *(_BYTE **)(*v4 + 24);
  if ( *(_BYTE **)(*v4 + 16) == v30 )
  {
    sub_16E7EE0(v29, "|", 1u);
  }
  else
  {
    *v30 = 124;
    ++*(_QWORD *)(v29 + 24);
  }
  v31 = *v4;
  v32 = *(_BYTE **)(*v4 + 24);
  if ( *(_BYTE **)(*v4 + 16) == v32 )
  {
    v31 = sub_16E7EE0(*v4, "{", 1u);
    if ( v68 != v66 )
LABEL_62:
      sub_16E7BA0((__int64 *)&v65);
  }
  else
  {
    *v32 = 123;
    ++*(_QWORD *)(v31 + 24);
    if ( v68 != v66 )
      goto LABEL_62;
  }
  v33 = sub_16E7EE0(v31, *v70, (size_t)v70[1]);
  v34 = *(_BYTE **)(v33 + 24);
  if ( *(_BYTE **)(v33 + 16) == v34 )
  {
    sub_16E7EE0(v33, "}", 1u);
  }
  else
  {
    *v34 = 125;
    ++*(_QWORD *)(v33 + 24);
  }
LABEL_41:
  v21 = *v4;
  v22 = *(_QWORD *)(*v4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(*v4 + 16) - v22) <= 4 )
  {
    sub_16E7EE0(v21, "}\"];\n", 5u);
  }
  else
  {
    *(_DWORD *)v22 = 995959421;
    *(_BYTE *)(v22 + 4) = 10;
    *(_QWORD *)(v21 + 24) += 5LL;
  }
  v23 = *(__int64 **)(v3 + 88);
  v24 = *(__int64 **)(v3 + 96);
  v25 = 0;
  if ( v24 != v23 )
  {
    while ( 1 )
    {
      sub_1DDF400((__int64)v4, v3, v25++, v23++);
      if ( v24 == v23 )
        break;
      if ( v25 == 64 )
      {
        do
          sub_1DDF400((__int64)v4, v3, 64, v23++);
        while ( v24 != v23 );
        break;
      }
    }
  }
  result = sub_16E7BC0((__int64 *)&v65);
  if ( (_QWORD *)v57[0] != v58 )
    result = (void *)j_j___libc_free_0(v57[0], v58[0] + 1LL);
  if ( v54 != (char *)v56 )
    return (void *)j_j___libc_free_0(v54, v56[0] + 1LL);
  return result;
}
