// Function: sub_1044070
// Address: 0x1044070
//
void *__fastcall sub_1044070(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rax
  __m128i si128; // xmm0
  unsigned __int8 *v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  size_t v14; // rdx
  unsigned __int64 v15; // rax
  unsigned int v16; // edx
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  unsigned int v26; // ebx
  __int64 v27; // rax
  __int64 v28; // rdi
  _QWORD *v29; // rdx
  unsigned __int64 v30; // r12
  size_t v31; // rax
  int v32; // esi
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rax
  void *result; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned int v41; // ebx
  unsigned __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // r12
  size_t v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // [rsp+10h] [rbp-100h]
  size_t v52; // [rsp+20h] [rbp-F0h]
  size_t v54; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v55; // [rsp+30h] [rbp-E0h]
  __int64 v56; // [rsp+30h] [rbp-E0h]
  __int64 *v57; // [rsp+30h] [rbp-E0h]
  int v58; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v59; // [rsp+40h] [rbp-D0h] BYREF
  size_t v60; // [rsp+48h] [rbp-C8h]
  _QWORD v61[2]; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int8 *v62; // [rsp+60h] [rbp-B0h] BYREF
  size_t v63; // [rsp+68h] [rbp-A8h]
  _QWORD v64[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+80h] [rbp-90h] BYREF
  size_t v66; // [rsp+88h] [rbp-88h]
  _QWORD v67[2]; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int8 *v68; // [rsp+A0h] [rbp-70h] BYREF
  size_t v69; // [rsp+A8h] [rbp-68h]
  _QWORD v70[12]; // [rsp+B0h] [rbp-60h] BYREF

  v65 = **(_QWORD **)(a1 + 8);
  sub_11F8430(
    (unsigned int)&v68,
    a2,
    0,
    (unsigned int)sub_103AAB0,
    (unsigned int)&v65,
    a6,
    (__int64)sub_103B590,
    (__int64)&v62);
  v59 = (unsigned __int8 *)v61;
  if ( sub_22417D0(&v68, 59, 0) == -1 )
  {
    v11 = (unsigned __int8 *)v61;
    v10 = 0;
  }
  else
  {
    v65 = 33;
    v7 = sub_22409D0(&v59, &v65, 0);
    v59 = (unsigned __int8 *)v7;
    v61[0] = v65;
    *(__m128i *)v7 = _mm_load_si128((const __m128i *)&xmmword_3F8E520);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E530);
    v9 = v59;
    *(_BYTE *)(v7 + 32) = 107;
    *(__m128i *)(v7 + 16) = si128;
    v10 = v65;
    v11 = &v9[v65];
  }
  v60 = v10;
  *v11 = 0;
  if ( v68 != (unsigned __int8 *)v70 )
    j_j___libc_free_0(v68, v70[0] + 1LL);
  v12 = sub_904010(*(_QWORD *)a1, "\tNode");
  v13 = sub_CB5A80(v12, a2);
  sub_904010(v13, " [shape=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_904010(*(_QWORD *)a1, "none,");
    v14 = v60;
    if ( !v60 )
      goto LABEL_7;
  }
  else
  {
    sub_904010(*(_QWORD *)a1, "record,");
    v14 = v60;
    if ( !v60 )
      goto LABEL_7;
  }
  v35 = sub_CB6200(*(_QWORD *)a1, v59, v14);
  sub_904010(v35, ",");
LABEL_7:
  sub_904010(*(_QWORD *)a1, "label=");
  if ( *(_BYTE *)(a1 + 16) )
  {
    v15 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a2 + 48 == v15 )
      goto LABEL_78;
    if ( !v15 )
      goto LABEL_65;
    if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA || (v16 = sub_B46E30(v15 - 24)) == 0 )
    {
LABEL_78:
      v18 = 1;
    }
    else
    {
      v17 = 0;
      do
      {
        if ( v16 == ++v17 )
        {
          v18 = v16;
          goto LABEL_16;
        }
      }
      while ( v17 != 64 );
      v18 = 65;
    }
LABEL_16:
    v55 = v18;
    v19 = sub_904010(*(_QWORD *)a1, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"");
    v20 = sub_904010(v19, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"");
    v21 = sub_CB59D0(v20, v55);
    sub_904010(v21, "\">");
  }
  else
  {
    sub_904010(*(_QWORD *)a1, "\"{");
  }
  v22 = *(_QWORD *)a1;
  v56 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v65 = **(_QWORD **)(a1 + 8);
    sub_11F8430(
      (unsigned int)&v68,
      a2,
      0,
      (unsigned int)sub_103AAB0,
      (unsigned int)&v65,
      v22,
      (__int64)sub_103B590,
      (__int64)&v62);
    v23 = sub_CB6200(v56, v68, v69);
    sub_904010(v23, "</td>");
    if ( v68 != (unsigned __int8 *)v70 )
      j_j___libc_free_0(v68, v70[0] + 1LL);
  }
  else
  {
    v68 = **(unsigned __int8 ***)(a1 + 8);
    sub_11F8430(
      (unsigned int)&v65,
      a2,
      0,
      (unsigned int)sub_103AAB0,
      (unsigned int)&v68,
      v22,
      (__int64)sub_103B590,
      (__int64)&v62);
    sub_C67200((__int64 *)&v68, (__int64)&v65);
    sub_CB6200(v56, v68, v69);
    if ( v68 != (unsigned __int8 *)v70 )
      j_j___libc_free_0(v68, v70[0] + 1LL);
    if ( (_QWORD *)v65 != v67 )
      j_j___libc_free_0(v65, v67[0] + 1LL);
  }
  v70[4] = &v62;
  v62 = (unsigned __int8 *)v64;
  v70[3] = 0x100000000LL;
  v63 = 0;
  LOBYTE(v64[0]) = 0;
  v69 = 0;
  memset(v70, 0, 24);
  v68 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v68, 0, 0, 0);
  if ( (unsigned __int8)sub_1043C80(a1, (__int64)&v68, a2) )
  {
    if ( *(_BYTE *)(a1 + 16) || (sub_904010(*(_QWORD *)a1, "|"), *(_BYTE *)(a1 + 16)) )
    {
      sub_CB6200(*(_QWORD *)a1, v62, v63);
    }
    else
    {
      v39 = sub_904010(*(_QWORD *)a1, "{");
      v40 = sub_CB6200(v39, v62, v63);
      sub_904010(v40, "}");
    }
  }
  v24 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 16) )
    sub_904010(v24, "</tr></table>>");
  else
    sub_904010(v24, "}\"");
  sub_904010(*(_QWORD *)a1, "];\n");
  v25 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v25 )
    goto LABEL_51;
  if ( !v25 )
LABEL_65:
    BUG();
  v51 = v25 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 <= 0xA )
  {
    v58 = sub_B46E30(v51);
    if ( v58 )
    {
      v57 = (__int64 *)a1;
      v26 = 0;
      while ( 1 )
      {
        v30 = sub_B46EC0(v51, v26);
        if ( v30 )
        {
          sub_103B6D0((__int64)&v65, a2, v26);
          v31 = v66;
          if ( (_QWORD *)v65 != v67 )
          {
            v52 = v66;
            j_j___libc_free_0(v65, v67[0] + 1LL);
            v31 = v52;
          }
          v32 = -1;
          v65 = (__int64)v67;
          if ( v31 )
            v32 = v26;
          sub_103ABA0(&v65, byte_3F871B3, (__int64)byte_3F871B3);
          v33 = *(_QWORD *)a1;
          v34 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v34) > 4 )
          {
            *(_DWORD *)v34 = 1685016073;
            *(_BYTE *)(v34 + 4) = 101;
            *(_QWORD *)(v33 + 32) += 5LL;
          }
          else
          {
            v33 = sub_CB6200(v33, "\tNode", 5u);
          }
          sub_CB5A80(v33, a2);
          if ( v32 != -1 )
          {
            v27 = sub_904010(*(_QWORD *)a1, ":s");
            sub_CB59F0(v27, v32);
          }
          v28 = *(_QWORD *)a1;
          v29 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
          if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v29 <= 7u )
          {
            v28 = sub_CB6200(v28, " -> Node", 8u);
          }
          else
          {
            *v29 = 0x65646F4E203E2D20LL;
            *(_QWORD *)(v28 + 32) += 8LL;
          }
          sub_CB5A80(v28, v30);
          if ( v66 )
          {
            v37 = sub_904010(*(_QWORD *)a1, "[");
            v38 = sub_CB6200(v37, (unsigned __int8 *)v65, v66);
            sub_904010(v38, "]");
          }
          sub_904010(*(_QWORD *)a1, ";\n");
          if ( (_QWORD *)v65 != v67 )
            j_j___libc_free_0(v65, v67[0] + 1LL);
        }
        if ( ++v26 == v58 )
          break;
        if ( v26 == 64 )
        {
          v41 = 64;
          v42 = a2;
          do
          {
            v46 = sub_B46EC0(v51, v41);
            if ( v46 )
            {
              sub_103B6D0((__int64)&v65, v42, v41);
              v47 = v66;
              if ( (_QWORD *)v65 != v67 )
              {
                v54 = v66;
                j_j___libc_free_0(v65, v67[0] + 1LL);
                v47 = v54;
              }
              v65 = (__int64)v67;
              if ( v47 )
              {
                sub_103ABA0(&v65, byte_3F871B3, (__int64)byte_3F871B3);
                v43 = sub_904010(*v57, "\tNode");
                sub_CB5A80(v43, v42);
                v44 = sub_904010(*v57, ":s");
                sub_CB59F0(v44, 64);
              }
              else
              {
                sub_103ABA0(&v65, byte_3F871B3, (__int64)byte_3F871B3);
                v48 = sub_904010(*v57, "\tNode");
                sub_CB5A80(v48, v42);
              }
              v45 = sub_904010(*v57, " -> Node");
              sub_CB5A80(v45, v46);
              if ( v66 )
              {
                v49 = sub_904010(*v57, "[");
                v50 = sub_CB6200(v49, (unsigned __int8 *)v65, v66);
                sub_904010(v50, "]");
              }
              sub_904010(*v57, ";\n");
              if ( (_QWORD *)v65 != v67 )
                j_j___libc_free_0(v65, v67[0] + 1LL);
            }
            ++v41;
          }
          while ( v58 != v41 );
          break;
        }
      }
    }
  }
LABEL_51:
  v68 = (unsigned __int8 *)&unk_49DD210;
  result = sub_CB5840((__int64)&v68);
  if ( v62 != (unsigned __int8 *)v64 )
    result = (void *)j_j___libc_free_0(v62, v64[0] + 1LL);
  if ( v59 != (unsigned __int8 *)v61 )
    return (void *)j_j___libc_free_0(v59, v61[0] + 1LL);
  return result;
}
