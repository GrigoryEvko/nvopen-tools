// Function: sub_1C14BC0
// Address: 0x1c14bc0
//
__int64 __fastcall sub_1C14BC0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, unsigned int a5)
{
  unsigned __int64 v5; // r15
  unsigned int v7; // r12d
  unsigned int v8; // ebx
  __int64 result; // rax
  void *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  unsigned int v18; // r9d
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i si128; // xmm0
  void *v25; // rdx
  __int64 v26; // rax
  _WORD *v27; // rdx
  _QWORD *v28; // rax
  __m128i *v29; // rdx
  __int64 v30; // rdi
  __m128i v31; // xmm0
  __int64 v32; // rax
  __m128i *v33; // rdx
  __int64 v34; // rdi
  __m128i v35; // xmm0
  void *v36; // rdx
  __int64 v37; // rax
  _WORD *v38; // rdx
  _QWORD *v39; // rax
  void *v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // rax
  __m128i *v43; // rdx
  __int64 v44; // rdi
  __m128i v45; // xmm0
  void *v46; // rdx
  __int64 v47; // rax
  _WORD *v48; // rdx
  void *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  void *v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rax
  void *v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  void *v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  void *v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  void *v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  void *v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  unsigned int v90; // [rsp+8h] [rbp-48h]
  unsigned __int64 v91; // [rsp+10h] [rbp-40h]

  v5 = HIDWORD(a4);
  v7 = a3;
  v8 = a4;
  v91 = HIDWORD(a3);
  result = 0;
  if ( (_DWORD)a1 )
  {
    if ( (_DWORD)a1 == 1 )
    {
      if ( HIDWORD(a1) <= 0x41 )
        goto LABEL_4;
      v56 = sub_16E8CB0();
      v57 = sub_1263B40((__int64)v56, "minor Version (");
      v58 = sub_16E7A90(v57, HIDWORD(a1));
      v59 = sub_1263B40(v58, ") newer than tool ");
      v60 = sub_1263B40(v59, "(should be ");
      v61 = sub_16E7AB0(v60, 65);
      sub_1263B40(v61, ")\n");
    }
    else
    {
      v90 = a1;
      v17 = sub_16E8CB0();
      v18 = a1;
      v19 = v17[3];
      v20 = (__int64)v17;
      if ( (unsigned __int64)(v17[2] - v19) <= 8 )
      {
        v80 = sub_16E7EE0((__int64)v17, "Version (", 9u);
        v18 = v90;
        v20 = v80;
      }
      else
      {
        *(_BYTE *)(v19 + 8) = 40;
        *(_QWORD *)v19 = 0x206E6F6973726556LL;
        v17[3] += 9LL;
      }
      v21 = sub_16E7A90(v20, v18);
      v22 = *(__m128i **)(v21 + 24);
      v23 = v21;
      if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 0x10u )
      {
        v81 = sub_16E7EE0(v21, ") not compatible ", 0x11u);
        v25 = *(void **)(v81 + 24);
        v23 = v81;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
        v22[1].m128i_i8[0] = 32;
        *v22 = si128;
        v25 = (void *)(*(_QWORD *)(v21 + 24) + 17LL);
        *(_QWORD *)(v21 + 24) = v25;
      }
      if ( *(_QWORD *)(v23 + 16) - (_QWORD)v25 <= 0xAu )
      {
        v23 = sub_16E7EE0(v23, "(should be ", 0xBu);
      }
      else
      {
        qmemcpy(v25, "(should be ", 11);
        *(_QWORD *)(v23 + 24) += 11LL;
      }
      v26 = sub_16E7AB0(v23, 1);
      v27 = *(_WORD **)(v26 + 24);
      if ( *(_QWORD *)(v26 + 16) - (_QWORD)v27 <= 1u )
      {
        sub_16E7EE0(v26, ")\n", 2u);
      }
      else
      {
        *v27 = 2601;
        *(_QWORD *)(v26 + 24) += 2LL;
      }
    }
    result = 1;
  }
LABEL_4:
  if ( !(_DWORD)a2 )
    goto LABEL_7;
  if ( (_DWORD)a2 == 2 )
  {
    if ( HIDWORD(a2) <= 0x62 )
      goto LABEL_7;
    v62 = sub_16E8CB0();
    v63 = sub_1263B40((__int64)v62, "minor NvvmIRVersion (");
    v64 = sub_16E7A90(v63, HIDWORD(a2));
    v65 = sub_1263B40(v64, ") newer than tool ");
    v66 = sub_1263B40(v65, "(should be ");
    v67 = sub_16E7AB0(v66, 98);
    sub_1263B40(v67, ")\n");
  }
  else
  {
    v39 = sub_16E8CB0();
    v40 = (void *)v39[3];
    v41 = (__int64)v39;
    if ( v39[2] - (_QWORD)v40 <= 0xEu )
    {
      v41 = sub_16E7EE0((__int64)v39, "NvvmIRVersion (", 0xFu);
    }
    else
    {
      qmemcpy(v40, "NvvmIRVersion (", 15);
      v39[3] += 15LL;
    }
    v42 = sub_16E7A90(v41, (unsigned int)a2);
    v43 = *(__m128i **)(v42 + 24);
    v44 = v42;
    if ( *(_QWORD *)(v42 + 16) - (_QWORD)v43 <= 0x10u )
    {
      v82 = sub_16E7EE0(v42, ") not compatible ", 0x11u);
      v46 = *(void **)(v82 + 24);
      v44 = v82;
    }
    else
    {
      v45 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v43[1].m128i_i8[0] = 32;
      *v43 = v45;
      v46 = (void *)(*(_QWORD *)(v42 + 24) + 17LL);
      *(_QWORD *)(v42 + 24) = v46;
    }
    if ( *(_QWORD *)(v44 + 16) - (_QWORD)v46 <= 0xAu )
    {
      v44 = sub_16E7EE0(v44, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v46, "(should be ", 11);
      *(_QWORD *)(v44 + 24) += 11LL;
    }
    v47 = sub_16E7AB0(v44, 2);
    v48 = *(_WORD **)(v47 + 24);
    if ( *(_QWORD *)(v47 + 16) - (_QWORD)v48 <= 1u )
    {
      sub_16E7EE0(v47, ")\n", 2u);
    }
    else
    {
      *v48 = 2601;
      *(_QWORD *)(v47 + 24) += 2LL;
    }
  }
  result = 1;
LABEL_7:
  if ( !v7 )
    goto LABEL_10;
  if ( v7 == 3 )
  {
    if ( (unsigned int)v91 <= 2 )
      goto LABEL_10;
    v68 = sub_16E8CB0();
    v69 = sub_1263B40((__int64)v68, "minor NvvmDebugVersion (");
    v70 = sub_16E7A90(v69, (unsigned int)v91);
    v71 = sub_1263B40(v70, ") newer than tool ");
    v72 = sub_1263B40(v71, "(should be ");
    v73 = sub_16E7AB0(v72, 2);
    sub_1263B40(v73, ")\n");
  }
  else
  {
    v28 = sub_16E8CB0();
    v29 = (__m128i *)v28[3];
    v30 = (__int64)v28;
    if ( v28[2] - (_QWORD)v29 <= 0x11u )
    {
      v30 = sub_16E7EE0((__int64)v28, "NvvmDebugVersion (", 0x12u);
    }
    else
    {
      v31 = _mm_load_si128((const __m128i *)&xmmword_3F6E0F0);
      v29[1].m128i_i16[0] = 10272;
      *v29 = v31;
      v28[3] += 18LL;
    }
    v32 = sub_16E7A90(v30, v7);
    v33 = *(__m128i **)(v32 + 24);
    v34 = v32;
    if ( *(_QWORD *)(v32 + 16) - (_QWORD)v33 <= 0x10u )
    {
      v83 = sub_16E7EE0(v32, ") not compatible ", 0x11u);
      v36 = *(void **)(v83 + 24);
      v34 = v83;
    }
    else
    {
      v35 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v33[1].m128i_i8[0] = 32;
      *v33 = v35;
      v36 = (void *)(*(_QWORD *)(v32 + 24) + 17LL);
      *(_QWORD *)(v32 + 24) = v36;
    }
    if ( *(_QWORD *)(v34 + 16) - (_QWORD)v36 <= 0xAu )
    {
      v34 = sub_16E7EE0(v34, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v36, "(should be ", 11);
      *(_QWORD *)(v34 + 24) += 11LL;
    }
    v37 = sub_16E7AB0(v34, 3);
    v38 = *(_WORD **)(v37 + 24);
    if ( *(_QWORD *)(v37 + 16) - (_QWORD)v38 <= 1u )
    {
      sub_16E7EE0(v37, ")\n", 2u);
    }
    else
    {
      *v38 = 2601;
      *(_QWORD *)(v37 + 24) += 2LL;
    }
  }
  result = 1;
LABEL_10:
  if ( v8 )
  {
    if ( (_BYTE)a5 )
    {
      if ( v8 > 7 )
      {
        v74 = sub_16E8CB0();
        v75 = sub_1263B40((__int64)v74, "LlvmVersion (");
        v76 = sub_16E7A90(v75, v8);
        v77 = sub_1263B40(v76, ") not compatible ");
        v78 = sub_1263B40(v77, "(should be ");
        v79 = sub_16E7AB0(v78, 7);
        sub_1263B40(v79, ")\n");
        return a5;
      }
      else if ( (_DWORD)v5 != 0 && v8 == 7 )
      {
        v84 = sub_16E8CB0();
        v85 = sub_1263B40((__int64)v84, "minor LlvmVersion (");
        v86 = sub_16E7A90(v85, (unsigned int)v5);
        v87 = sub_1263B40(v86, ") newer than tool ");
        v88 = sub_1263B40(v87, "(should be ");
        v89 = sub_16E7AB0(v88, 0);
        sub_1263B40(v89, ")\n");
        return ((_DWORD)v5 != 0) & (unsigned __int8)(v8 == 7);
      }
    }
    else if ( v8 == 7 )
    {
      if ( (_DWORD)v5 )
      {
        v10 = sub_16E8CB0();
        v11 = sub_1263B40((__int64)v10, "minor LlvmVersion (");
        v12 = sub_16E7A90(v11, (unsigned int)v5);
        v13 = sub_1263B40(v12, ") newer than tool ");
        v14 = sub_1263B40(v13, "(should be ");
        v15 = sub_16E7AB0(v14, 0);
        v16 = sub_1263B40(v15, ").");
        sub_1263B40(v16, " Must be same as tool for ascii dumps\n");
        return 1;
      }
    }
    else
    {
      v49 = sub_16E8CB0();
      v50 = sub_1263B40((__int64)v49, "LlvmVersion (");
      v51 = sub_16E7A90(v50, v8);
      v52 = sub_1263B40(v51, ") not compatible ");
      v53 = *(void **)(v52 + 24);
      v54 = v52;
      if ( *(_QWORD *)(v52 + 16) - (_QWORD)v53 <= 0xAu )
      {
        v54 = sub_16E7EE0(v52, "(should be ", 0xBu);
      }
      else
      {
        qmemcpy(v53, "(should be ", 11);
        *(_QWORD *)(v52 + 24) += 11LL;
      }
      v55 = sub_16E7AB0(v54, 7);
      sub_1263B40(v55, ")\n");
      return 1;
    }
  }
  return result;
}
