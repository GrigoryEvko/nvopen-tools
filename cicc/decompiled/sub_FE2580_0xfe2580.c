// Function: sub_FE2580
// Address: 0xfe2580
//
void *__fastcall sub_FE2580(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 *v4; // r15
  unsigned int v5; // r13d
  unsigned __int64 v6; // r14
  unsigned int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  size_t v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned int v21; // edx
  int v22; // eax
  unsigned __int64 v23; // r13
  __int64 v24; // rdi
  __m128i *v25; // rdx
  __m128i si128; // xmm0
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __int64 v29; // rax
  _WORD *v30; // rdx
  __int64 v31; // r14
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdi
  _WORD *v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // r14
  unsigned int v42; // r15d
  unsigned __int64 v43; // r13
  __int64 v44; // r12
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  int v47; // edx
  void *result; // rax
  void *v49; // rdx
  _BYTE *v50; // rax
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rbx
  __int64 v54; // r14
  __int64 v55; // rsi
  unsigned __int64 v56; // rax
  __int64 v57; // rdi
  _WORD *v58; // rdx
  __int64 v59; // r8
  _BYTE *v60; // rax
  __int64 v61; // r9
  _BYTE *v62; // rax
  unsigned __int64 v63; // rax
  __int64 v64; // rbx
  unsigned __int64 v65; // r12
  unsigned __int64 v66; // r13
  unsigned __int64 v67; // rax
  __int64 v68; // r12
  unsigned int v69; // ebx
  unsigned __int64 v70; // r15
  __int64 v71; // rdx
  __int64 v72; // rdi
  _BYTE *v73; // rax
  __int64 v74; // rdi
  _BYTE *v75; // rax
  __int64 v76; // rdi
  _BYTE *v77; // rax
  __int64 v78; // rax
  __int64 v79; // [rsp+10h] [rbp-F0h]
  __int64 v80; // [rsp+10h] [rbp-F0h]
  int v81; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v83; // [rsp+30h] [rbp-D0h] BYREF
  size_t v84; // [rsp+38h] [rbp-C8h]
  _QWORD v85[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v86[2]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD v87[2]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int8 *v88; // [rsp+70h] [rbp-90h] BYREF
  size_t v89; // [rsp+78h] [rbp-88h]
  _QWORD v90[2]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int8 *v91; // [rsp+90h] [rbp-70h] BYREF
  size_t v92; // [rsp+98h] [rbp-68h]
  _BYTE *v93; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v94; // [rsp+A8h] [rbp-58h]
  _BYTE *v95; // [rsp+B0h] [rbp-50h]
  __int64 v96; // [rsp+B8h] [rbp-48h]
  unsigned __int8 **v97; // [rsp+C0h] [rbp-40h]

  v2 = a2;
  v3 = (__int64 *)a1;
  v4 = **(__int64 ***)(a1 + 8);
  v84 = 0;
  LOBYTE(v85[0]) = 0;
  v5 = qword_4F8DE48[8];
  v83 = (unsigned __int8 *)v85;
  if ( LODWORD(qword_4F8DE48[8]) )
  {
    if ( !*(_QWORD *)(a1 + 32) )
    {
      v51 = *(_QWORD *)(sub_FDC440(v4) + 80);
      v52 = sub_FDC440(v4) + 72;
      if ( v51 != v52 )
      {
        v53 = v51;
        v54 = v52;
        do
        {
          v55 = v53 - 24;
          if ( !v53 )
            v55 = 0;
          v56 = sub_FDD860(v4, v55);
          if ( v56 < *(_QWORD *)(a1 + 32) )
            v56 = *(_QWORD *)(a1 + 32);
          *(_QWORD *)(a1 + 32) = v56;
          v53 = *(_QWORD *)(v53 + 8);
        }
        while ( v54 != v53 );
        v3 = (__int64 *)a1;
        v2 = a2;
      }
    }
    v6 = sub_FDD860(v4, v2);
    v7 = sub_F02DD0(v5, 0x64u);
    v91 = (unsigned __int8 *)v3[4];
    if ( sub_1098D20(&v91, v7) <= v6 )
    {
      v96 = 0x100000000LL;
      v92 = 0;
      v91 = (unsigned __int8 *)&unk_49DD210;
      v97 = &v83;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      sub_CB5980((__int64)&v91, 0, 0, 0);
      v49 = v95;
      if ( (unsigned __int64)(v94 - (_QWORD)v95) <= 0xA )
      {
        sub_CB6200((__int64)&v91, (unsigned __int8 *)"color=\"red\"", 0xBu);
        v50 = v95;
      }
      else
      {
        v95[10] = 34;
        qmemcpy(v49, "color=\"red", 10);
        v50 = v95 + 11;
        v95 += 11;
      }
      if ( v93 != v50 )
        sub_CB5AE0((__int64 *)&v91);
      v91 = (unsigned __int8 *)&unk_49DD210;
      sub_CB5840((__int64)&v91);
    }
  }
  v8 = *v3;
  v9 = *(_QWORD *)(*v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*v3 + 24) - v9) <= 4 )
  {
    v8 = sub_CB6200(v8, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v9 = 1685016073;
    *(_BYTE *)(v9 + 4) = 101;
    *(_QWORD *)(v8 + 32) += 5LL;
  }
  v10 = sub_CB5A80(v8, v2);
  v11 = *(_QWORD **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 7u )
  {
    sub_CB6200(v10, " [shape=", 8u);
  }
  else
  {
    *v11 = 0x3D65706168735B20LL;
    *(_QWORD *)(v10 + 32) += 8LL;
  }
  v12 = *v3;
  v13 = *(_QWORD *)(*v3 + 32);
  v14 = *(_QWORD *)(*v3 + 24) - v13;
  if ( *((_BYTE *)v3 + 16) )
  {
    if ( v14 > 4 )
    {
      *(_DWORD *)v13 = 1701736302;
      *(_BYTE *)(v13 + 4) = 44;
      v15 = v84;
      *(_QWORD *)(v12 + 32) += 5LL;
      if ( v15 )
        goto LABEL_11;
      goto LABEL_16;
    }
    sub_CB6200(v12, (unsigned __int8 *)"none,", 5u);
  }
  else if ( v14 <= 6 )
  {
    sub_CB6200(v12, (unsigned __int8 *)"record,", 7u);
  }
  else
  {
    *(_DWORD *)v13 = 1868785010;
    *(_WORD *)(v13 + 4) = 25714;
    *(_BYTE *)(v13 + 6) = 44;
    *(_QWORD *)(v12 + 32) += 7LL;
  }
  v15 = v84;
  if ( v84 )
  {
LABEL_11:
    v16 = sub_CB6200(*v3, v83, v15);
    v17 = *(_BYTE **)(v16 + 32);
    if ( *(_BYTE **)(v16 + 24) == v17 )
    {
      sub_CB6200(v16, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v17 = 44;
      ++*(_QWORD *)(v16 + 32);
    }
  }
LABEL_16:
  v18 = *v3;
  v19 = *(_QWORD *)(*v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*v3 + 24) - v19) <= 5 )
  {
    sub_CB6200(v18, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v19 = 1700946284;
    *(_WORD *)(v19 + 4) = 15724;
    *(_QWORD *)(v18 + 32) += 6LL;
  }
  if ( *((_BYTE *)v3 + 16) )
  {
    v20 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v20 == v2 + 48 )
      goto LABEL_119;
    if ( !v20 )
      goto LABEL_123;
    if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA || (v21 = sub_B46E30(v20 - 24)) == 0 )
    {
LABEL_119:
      v23 = 1;
    }
    else
    {
      v22 = 0;
      do
      {
        if ( v21 == ++v22 )
        {
          v23 = v21;
          goto LABEL_27;
        }
      }
      while ( v22 != 64 );
      v23 = 65;
    }
LABEL_27:
    v24 = *v3;
    v25 = *(__m128i **)(*v3 + 32);
    if ( *(_QWORD *)(*v3 + 24) - (_QWORD)v25 <= 0x30u )
    {
      v78 = sub_CB6200(v24, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v27 = *(_QWORD *)(v78 + 32);
      v24 = v78;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v25[3].m128i_i8[0] = 34;
      *v25 = si128;
      v25[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v25[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v27 = *(_QWORD *)(v24 + 32) + 49LL;
      *(_QWORD *)(v24 + 32) = v27;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v27) <= 0x2E )
    {
      v24 = sub_CB6200(v24, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v28 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v27 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v27 = v28;
      *(__m128i *)(v27 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v24 + 32) += 47LL;
    }
    v29 = sub_CB59D0(v24, v23);
    v30 = *(_WORD **)(v29 + 32);
    if ( *(_QWORD *)(v29 + 24) - (_QWORD)v30 <= 1u )
    {
      sub_CB6200(v29, "\">", 2u);
    }
    else
    {
      *v30 = 15906;
      *(_QWORD *)(v29 + 32) += 2LL;
    }
  }
  else
  {
    v57 = *v3;
    v58 = *(_WORD **)(*v3 + 32);
    if ( *(_QWORD *)(*v3 + 24) - (_QWORD)v58 <= 1u )
    {
      sub_CB6200(v57, (unsigned __int8 *)"\"{", 2u);
    }
    else
    {
      *v58 = 31522;
      *(_QWORD *)(v57 + 32) += 2LL;
    }
  }
  v31 = *v3;
  v32 = *(__int64 **)v3[1];
  if ( *((_BYTE *)v3 + 16) )
  {
    sub_FDD8D0((__int64)&v91, v2, v32);
    v33 = sub_CB6200(v31, v91, v92);
    v34 = *(_QWORD *)(v33 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v33 + 24) - v34) <= 4 )
    {
      sub_CB6200(v33, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v34 = 1685335868;
      *(_BYTE *)(v34 + 4) = 62;
      *(_QWORD *)(v33 + 32) += 5LL;
    }
    if ( v91 != (unsigned __int8 *)&v93 )
      j_j___libc_free_0(v91, v93 + 1);
  }
  else
  {
    sub_FDD8D0((__int64)&v88, v2, v32);
    sub_C67200((__int64 *)&v91, (__int64)&v88);
    sub_CB6200(v31, v91, v92);
    if ( v91 != (unsigned __int8 *)&v93 )
      j_j___libc_free_0(v91, v93 + 1);
    if ( v88 != (unsigned __int8 *)v90 )
      j_j___libc_free_0(v88, v90[0] + 1LL);
  }
  v86[0] = (__int64)v87;
  sub_FDB1F0(v86, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v86[1] )
  {
    v61 = *v3;
    v62 = *(_BYTE **)(*v3 + 32);
    if ( *(_BYTE **)(*v3 + 24) == v62 )
    {
      v61 = sub_CB6200(*v3, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v62 = 124;
      ++*(_QWORD *)(v61 + 32);
    }
    v80 = v61;
    sub_C67200((__int64 *)&v91, (__int64)v86);
    sub_CB6200(v80, v91, v92);
    if ( v91 != (unsigned __int8 *)&v93 )
      j_j___libc_free_0(v91, v93 + 1);
  }
  v88 = (unsigned __int8 *)v90;
  sub_FDB1F0((__int64 *)&v88, byte_3F871B3, (__int64)byte_3F871B3);
  if ( v89 )
  {
    v59 = *v3;
    v60 = *(_BYTE **)(*v3 + 32);
    if ( *(_BYTE **)(*v3 + 24) == v60 )
    {
      v59 = sub_CB6200(*v3, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v60 = 124;
      ++*(_QWORD *)(v59 + 32);
    }
    v79 = v59;
    sub_C67200((__int64 *)&v91, (__int64)&v88);
    sub_CB6200(v79, v91, v92);
    if ( v91 != (unsigned __int8 *)&v93 )
      j_j___libc_free_0(v91, v93 + 1);
  }
  if ( v88 != (unsigned __int8 *)v90 )
    j_j___libc_free_0(v88, v90[0] + 1LL);
  if ( (_QWORD *)v86[0] != v87 )
    j_j___libc_free_0(v86[0], v87[0] + 1LL);
  LOBYTE(v90[0]) = 0;
  v88 = (unsigned __int8 *)v90;
  v96 = 0x100000000LL;
  v89 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v91 = (unsigned __int8 *)&unk_49DD210;
  v97 = &v88;
  sub_CB5980((__int64)&v91, 0, 0, 0);
  if ( (unsigned __int8)sub_FE1BC0((__int64)v3, (__int64)&v91, v2) )
  {
    if ( *((_BYTE *)v3 + 16) )
      goto LABEL_46;
    v72 = *v3;
    v73 = *(_BYTE **)(*v3 + 32);
    if ( *(_BYTE **)(*v3 + 24) == v73 )
    {
      sub_CB6200(v72, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v73 = 124;
      ++*(_QWORD *)(v72 + 32);
    }
    v74 = *v3;
    if ( *((_BYTE *)v3 + 16) )
    {
LABEL_46:
      sub_CB6200(*v3, v88, v89);
    }
    else
    {
      v75 = *(_BYTE **)(v74 + 32);
      if ( *(_BYTE **)(v74 + 24) == v75 )
      {
        v74 = sub_CB6200(v74, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v75 = 123;
        ++*(_QWORD *)(v74 + 32);
      }
      v76 = sub_CB6200(v74, v88, v89);
      v77 = *(_BYTE **)(v76 + 32);
      if ( *(_BYTE **)(v76 + 24) == v77 )
      {
        sub_CB6200(v76, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v77 = 125;
        ++*(_QWORD *)(v76 + 32);
      }
    }
  }
  v35 = *v3;
  v36 = *(_WORD **)(*v3 + 32);
  v37 = *(_QWORD *)(*v3 + 24) - (_QWORD)v36;
  if ( *((_BYTE *)v3 + 16) )
  {
    if ( v37 <= 0xD )
    {
      sub_CB6200(v35, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v36, "</tr></table>>", 14);
      *(_QWORD *)(v35 + 32) += 14LL;
    }
  }
  else if ( v37 <= 1 )
  {
    sub_CB6200(v35, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v36 = 8829;
    *(_QWORD *)(v35 + 32) += 2LL;
  }
  v38 = *v3;
  v39 = *(_QWORD *)(*v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*v3 + 24) - v39) <= 2 )
  {
    sub_CB6200(v38, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v39 + 2) = 10;
    *(_WORD *)v39 = 15197;
    *(_QWORD *)(v38 + 32) += 3LL;
  }
  v40 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v40 == v2 + 48 )
    goto LABEL_59;
  if ( !v40 )
LABEL_123:
    BUG();
  v41 = v40 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 <= 0xA )
  {
    v81 = sub_B46E30(v41);
    if ( v81 )
    {
      v42 = 0;
      v43 = v2;
      v44 = (__int64)v3;
      v45 = 0;
      while ( 1 )
      {
        v46 = v42;
        v47 = v42++;
        v45 = v46 | v45 & 0xFFFFFFFF00000000LL;
        sub_FE2040(v44, v43, v47, v41, v45);
        if ( v42 == v81 )
          break;
        if ( v42 == 64 )
        {
          v63 = v45;
          v64 = v44;
          v65 = v43;
          v66 = v63;
          v67 = v65;
          v68 = v64;
          v69 = 64;
          v70 = v67;
          do
          {
            v71 = v69++;
            v66 = v71 | v66 & 0xFFFFFFFF00000000LL;
            sub_FE2040(v68, v70, 64, v41, v66);
          }
          while ( v81 != v69 );
          break;
        }
      }
    }
  }
LABEL_59:
  v91 = (unsigned __int8 *)&unk_49DD210;
  result = sub_CB5840((__int64)&v91);
  if ( v88 != (unsigned __int8 *)v90 )
    result = (void *)j_j___libc_free_0(v88, v90[0] + 1LL);
  if ( v83 != (unsigned __int8 *)v85 )
    return (void *)j_j___libc_free_0(v83, v85[0] + 1LL);
  return result;
}
