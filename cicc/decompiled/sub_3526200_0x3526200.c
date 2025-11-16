// Function: sub_3526200
// Address: 0x3526200
//
void __fastcall sub_3526200(_BYTE *a1, unsigned __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdi
  __m128i *v13; // r8
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // esi
  int v18; // ebx
  unsigned int v19; // ebx
  __m128i si128; // xmm0
  __int64 v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rax
  _WORD *v24; // rdx
  __int64 v25; // r13
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int8 *v29; // rdi
  __int64 v30; // rdi
  _WORD *v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  int v35; // r13d
  unsigned __int64 *v36; // rbx
  int v37; // ecx
  unsigned __int64 v38; // r15
  size_t v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  int v42; // ecx
  __int64 v43; // rdi
  _QWORD *v44; // rdx
  __int64 v45; // rdi
  _WORD *v46; // rdx
  __int64 v47; // rdi
  _QWORD *v48; // rdx
  __int64 v49; // rdi
  _WORD *v50; // rdx
  unsigned __int64 v51; // r14
  size_t v52; // r15
  int v53; // r15d
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rdi
  _WORD *v57; // rdx
  __int64 v58; // rdi
  _WORD *v59; // rdx
  __int64 v60; // rdi
  _BYTE *v61; // rax
  __int64 v62; // rdi
  _BYTE *v63; // rax
  __int64 v64; // rdi
  _BYTE *v65; // rax
  __int64 v66; // rdi
  _BYTE *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdi
  _BYTE *v72; // rax
  __int64 v73; // rdi
  _BYTE *v74; // rax
  __int64 v75; // rdi
  _BYTE *v76; // rax
  size_t v77; // [rsp+20h] [rbp-F0h]
  int v78; // [rsp+20h] [rbp-F0h]
  int v79; // [rsp+20h] [rbp-F0h]
  unsigned __int64 *v81; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v82; // [rsp+60h] [rbp-B0h] BYREF
  size_t v83; // [rsp+68h] [rbp-A8h]
  _BYTE v84[16]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int8 *v85; // [rsp+80h] [rbp-90h] BYREF
  size_t v86; // [rsp+88h] [rbp-88h]
  _QWORD v87[2]; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int8 *v88; // [rsp+A0h] [rbp-70h] BYREF
  size_t v89; // [rsp+A8h] [rbp-68h]
  _QWORD v90[12]; // [rsp+B0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 4 )
  {
    v3 = sub_CB6200(v3, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v4 = 1685016073;
    *(_BYTE *)(v4 + 4) = 101;
    *(_QWORD *)(v3 + 32) += 5LL;
  }
  v5 = sub_CB5A80(v3, a2);
  v6 = *(_QWORD **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 7u )
  {
    sub_CB6200(v5, " [shape=", 8u);
  }
  else
  {
    *v6 = 0x3D65706168735B20LL;
    *(_QWORD *)(v5 + 32) += 8LL;
  }
  v7 = *(_QWORD *)a1;
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - v8;
  if ( a1[16] )
  {
    if ( v9 <= 4 )
    {
      sub_CB6200(v7, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v8 = 1701736302;
      *(_BYTE *)(v8 + 4) = 44;
      *(_QWORD *)(v7 + 32) += 5LL;
    }
  }
  else if ( v9 <= 6 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"record,", 7u);
  }
  else
  {
    *(_DWORD *)v8 = 1868785010;
    *(_WORD *)(v8 + 4) = 25714;
    *(_BYTE *)(v8 + 6) = 44;
    *(_QWORD *)(v7 + 32) += 7LL;
  }
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v11) <= 5 )
  {
    sub_CB6200(v10, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v11 = 1700946284;
    *(_WORD *)(v11 + 4) = 15724;
    *(_QWORD *)(v10 + 32) += 6LL;
  }
  v12 = *(_QWORD *)a1;
  v13 = *(__m128i **)(*(_QWORD *)a1 + 32LL);
  v14 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v13;
  if ( a1[16] )
  {
    v15 = *(_QWORD *)(a2 + 112);
    v16 = v15 + 8LL * *(unsigned int *)(a2 + 120);
    if ( v15 == v16 )
    {
      v19 = 1;
    }
    else
    {
      v17 = 0;
      do
      {
        v15 += 8;
        ++v17;
      }
      while ( v15 != v16 && v17 != 64 );
      v18 = 1;
      if ( v17 )
        v18 = v17;
      v19 = (v15 != v16) + v18;
    }
    if ( v14 <= 0x30 )
    {
      v70 = sub_CB6200(v12, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v21 = *(_QWORD *)(v70 + 32);
      v12 = v70;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v13[3].m128i_i8[0] = 34;
      *v13 = si128;
      v13[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v13[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v21 = *(_QWORD *)(v12 + 32) + 49LL;
      *(_QWORD *)(v12 + 32) = v21;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v12 + 24) - v21) <= 0x2E )
    {
      v12 = sub_CB6200(v12, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v21 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v21 = v22;
      *(__m128i *)(v21 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v12 + 32) += 47LL;
    }
    v23 = sub_CB59D0(v12, v19);
    v24 = *(_WORD **)(v23 + 32);
    if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 1u )
    {
      sub_CB6200(v23, "\">", 2u);
    }
    else
    {
      *v24 = 15906;
      *(_QWORD *)(v23 + 32) += 2LL;
    }
  }
  else if ( v14 <= 1 )
  {
    sub_CB6200(v12, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    v13->m128i_i16[0] = 31522;
    *(_QWORD *)(v12 + 32) += 2LL;
  }
  v25 = *(_QWORD *)a1;
  v26 = a1[17];
  if ( a1[16] )
  {
    if ( v26 )
      sub_35254A0((__int64)&v88, a2);
    else
      sub_3525630(
        (__int64)&v88,
        a2,
        (void (__fastcall *)(__int64, _QWORD *, __int64))sub_3525190,
        (__int64)&v85,
        (void (__fastcall *)(__int64, __int64, unsigned int *, char *))sub_11F32A0,
        (__int64)sub_3525140);
    v27 = sub_CB6200(v25, v88, v89);
    v28 = *(_QWORD *)(v27 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v27 + 24) - v28) <= 4 )
    {
      sub_CB6200(v27, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v28 = 1685335868;
      *(_BYTE *)(v28 + 4) = 62;
      *(_QWORD *)(v27 + 32) += 5LL;
    }
    v29 = v88;
    if ( v88 != (unsigned __int8 *)v90 )
LABEL_32:
      j_j___libc_free_0((unsigned __int64)v29);
  }
  else
  {
    if ( v26 )
      sub_35254A0((__int64)&v85, a2);
    else
      sub_3525630(
        (__int64)&v85,
        a2,
        (void (__fastcall *)(__int64, _QWORD *, __int64))sub_3525190,
        (__int64)&v88,
        (void (__fastcall *)(__int64, __int64, unsigned int *, char *))sub_11F32A0,
        (__int64)sub_3525140);
    sub_C67200((__int64 *)&v88, (__int64)&v85);
    sub_CB6200(v25, v88, v89);
    if ( v88 != (unsigned __int8 *)v90 )
      j_j___libc_free_0((unsigned __int64)v88);
    v29 = v85;
    if ( v85 != (unsigned __int8 *)v87 )
      goto LABEL_32;
  }
  v82 = v84;
  v90[3] = 0x100000000LL;
  v83 = 0;
  v84[0] = 0;
  v88 = (unsigned __int8 *)&unk_49DD210;
  v89 = 0;
  memset(v90, 0, 24);
  v90[4] = &v82;
  sub_CB5980((__int64)&v88, 0, 0, 0);
  if ( (unsigned __int8)sub_3525DB0((__int64)a1, (__int64)&v88, a2) )
  {
    if ( a1[16] )
      goto LABEL_35;
    v71 = *(_QWORD *)a1;
    v72 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v72 )
    {
      sub_CB6200(v71, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v72 = 124;
      ++*(_QWORD *)(v71 + 32);
    }
    v73 = *(_QWORD *)a1;
    if ( a1[16] )
    {
LABEL_35:
      sub_CB6200(*(_QWORD *)a1, v82, v83);
    }
    else
    {
      v74 = *(_BYTE **)(v73 + 32);
      if ( *(_BYTE **)(v73 + 24) == v74 )
      {
        v73 = sub_CB6200(v73, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v74 = 123;
        ++*(_QWORD *)(v73 + 32);
      }
      v75 = sub_CB6200(v73, v82, v83);
      v76 = *(_BYTE **)(v75 + 32);
      if ( *(_BYTE **)(v75 + 24) == v76 )
      {
        sub_CB6200(v75, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v76 = 125;
        ++*(_QWORD *)(v75 + 32);
      }
    }
  }
  v30 = *(_QWORD *)a1;
  v31 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  v32 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v31;
  if ( a1[16] )
  {
    if ( v32 <= 0xD )
    {
      sub_CB6200(v30, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v31, "</tr></table>>", 14);
      *(_QWORD *)(v30 + 32) += 14LL;
    }
  }
  else if ( v32 <= 1 )
  {
    sub_CB6200(v30, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v31 = 8829;
    *(_QWORD *)(v30 + 32) += 2LL;
  }
  v33 = *(_QWORD *)a1;
  v34 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v34) <= 2 )
  {
    sub_CB6200(v33, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v34 + 2) = 10;
    *(_WORD *)v34 = 15197;
    *(_QWORD *)(v33 + 32) += 3LL;
  }
  v35 = 0;
  v36 = *(unsigned __int64 **)(a2 + 112);
  v81 = &v36[*(unsigned int *)(a2 + 120)];
  if ( v81 != v36 )
  {
    while ( 1 )
    {
      v38 = *v36;
      if ( *v36 )
        break;
LABEL_46:
      ++v36;
      ++v35;
      if ( v36 == v81 )
        goto LABEL_81;
      if ( v35 == 64 )
      {
        while ( v36 != v81 )
        {
          v51 = *v36;
          if ( *v36 )
          {
            v85 = (unsigned __int8 *)v87;
            sub_3525230((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
            v52 = v86;
            if ( v85 != (unsigned __int8 *)v87 )
              j_j___libc_free_0((unsigned __int64)v85);
            v85 = (unsigned __int8 *)v87;
            sub_3525230((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
            v53 = v52 == 0 ? -1 : 0x40;
            v54 = *(_QWORD *)a1;
            v55 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
            if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v55) > 4 )
            {
              *(_DWORD *)v55 = 1685016073;
              *(_BYTE *)(v55 + 4) = 101;
              *(_QWORD *)(v54 + 32) += 5LL;
            }
            else
            {
              v54 = sub_CB6200(v54, "\tNode", 5u);
            }
            sub_CB5A80(v54, a2);
            if ( v53 != -1 )
            {
              v58 = *(_QWORD *)a1;
              v59 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
              if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v59 <= 1u )
              {
                v58 = sub_CB6200(v58, ":s", 2u);
              }
              else
              {
                *v59 = 29498;
                *(_QWORD *)(v58 + 32) += 2LL;
              }
              sub_CB59F0(v58, 64);
            }
            v47 = *(_QWORD *)a1;
            v48 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v48 <= 7u )
            {
              v47 = sub_CB6200(v47, " -> Node", 8u);
            }
            else
            {
              *v48 = 0x65646F4E203E2D20LL;
              *(_QWORD *)(v47 + 32) += 8LL;
            }
            sub_CB5A80(v47, v51);
            if ( v86 )
            {
              v60 = *(_QWORD *)a1;
              v61 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
              if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v61 )
              {
                v60 = sub_CB6200(v60, (unsigned __int8 *)"[", 1u);
              }
              else
              {
                *v61 = 91;
                ++*(_QWORD *)(v60 + 32);
              }
              v62 = sub_CB6200(v60, v85, v86);
              v63 = *(_BYTE **)(v62 + 32);
              if ( *(_BYTE **)(v62 + 24) == v63 )
              {
                sub_CB6200(v62, (unsigned __int8 *)"]", 1u);
              }
              else
              {
                *v63 = 93;
                ++*(_QWORD *)(v62 + 32);
              }
            }
            v49 = *(_QWORD *)a1;
            v50 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v50 <= 1u )
            {
              sub_CB6200(v49, (unsigned __int8 *)";\n", 2u);
            }
            else
            {
              *v50 = 2619;
              *(_QWORD *)(v49 + 32) += 2LL;
            }
            if ( v85 != (unsigned __int8 *)v87 )
              j_j___libc_free_0((unsigned __int64)v85);
          }
          ++v36;
        }
        goto LABEL_81;
      }
    }
    v85 = (unsigned __int8 *)v87;
    sub_3525230((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
    v39 = v86;
    if ( v85 != (unsigned __int8 *)v87 )
    {
      v77 = v86;
      j_j___libc_free_0((unsigned __int64)v85);
      v39 = v77;
    }
    v85 = (unsigned __int8 *)v87;
    if ( v39 )
    {
      sub_3525230((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
      v37 = v35;
      if ( v35 > 64 )
      {
LABEL_44:
        if ( v85 != (unsigned __int8 *)v87 )
          j_j___libc_free_0((unsigned __int64)v85);
        goto LABEL_46;
      }
    }
    else
    {
      sub_3525230((__int64 *)&v85, byte_3F871B3, (__int64)byte_3F871B3);
      v37 = -1;
    }
    v40 = *(_QWORD *)a1;
    v41 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v41) <= 4 )
    {
      v79 = v37;
      v68 = sub_CB6200(v40, "\tNode", 5u);
      v37 = v79;
      v40 = v68;
    }
    else
    {
      *(_DWORD *)v41 = 1685016073;
      *(_BYTE *)(v41 + 4) = 101;
      *(_QWORD *)(v40 + 32) += 5LL;
    }
    v78 = v37;
    sub_CB5A80(v40, a2);
    v42 = v78;
    if ( v78 >= 0 )
    {
      v56 = *(_QWORD *)a1;
      v57 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
      if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v57 <= 1u )
      {
        v69 = sub_CB6200(v56, ":s", 2u);
        v42 = v78;
        v56 = v69;
      }
      else
      {
        *v57 = 29498;
        *(_QWORD *)(v56 + 32) += 2LL;
      }
      sub_CB59F0(v56, v42);
    }
    v43 = *(_QWORD *)a1;
    v44 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v44 <= 7u )
    {
      v43 = sub_CB6200(v43, " -> Node", 8u);
    }
    else
    {
      *v44 = 0x65646F4E203E2D20LL;
      *(_QWORD *)(v43 + 32) += 8LL;
    }
    sub_CB5A80(v43, v38);
    if ( v86 )
    {
      v64 = *(_QWORD *)a1;
      v65 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v65 )
      {
        v64 = sub_CB6200(v64, (unsigned __int8 *)"[", 1u);
      }
      else
      {
        *v65 = 91;
        ++*(_QWORD *)(v64 + 32);
      }
      v66 = sub_CB6200(v64, v85, v86);
      v67 = *(_BYTE **)(v66 + 32);
      if ( *(_BYTE **)(v66 + 24) == v67 )
      {
        sub_CB6200(v66, (unsigned __int8 *)"]", 1u);
      }
      else
      {
        *v67 = 93;
        ++*(_QWORD *)(v66 + 32);
      }
    }
    v45 = *(_QWORD *)a1;
    v46 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v46 <= 1u )
    {
      sub_CB6200(v45, (unsigned __int8 *)";\n", 2u);
    }
    else
    {
      *v46 = 2619;
      *(_QWORD *)(v45 + 32) += 2LL;
    }
    goto LABEL_44;
  }
LABEL_81:
  v88 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v88);
  if ( v82 != v84 )
    j_j___libc_free_0((unsigned __int64)v82);
}
