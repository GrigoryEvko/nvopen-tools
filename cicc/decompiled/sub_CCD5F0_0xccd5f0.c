// Function: sub_CCD5F0
// Address: 0xccd5f0
//
__int64 __fastcall sub_CCD5F0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned int a5)
{
  unsigned __int64 v5; // r15
  unsigned int v6; // r14d
  unsigned int v8; // r12d
  unsigned int v9; // ebx
  __int64 result; // rax
  void *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // rdi
  __m128i si128; // xmm0
  void *v26; // rdx
  signed __int64 v27; // rsi
  __int64 v28; // rax
  _QWORD *v29; // rax
  __m128i *v30; // rdx
  __int64 v31; // rdi
  __m128i v32; // xmm0
  __int64 v33; // rax
  __m128i *v34; // rdx
  __int64 v35; // rdi
  __m128i v36; // xmm0
  void *v37; // rdx
  __int64 v38; // rax
  _WORD *v39; // rdx
  _QWORD *v40; // rax
  void *v41; // rdx
  __int64 v42; // rdi
  __int64 v43; // rax
  __m128i *v44; // rdx
  __int64 v45; // rdi
  __m128i v46; // xmm0
  void *v47; // rdx
  __int64 v48; // rax
  _WORD *v49; // rdx
  void *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  void *v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  void *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  void *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  void *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  void *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned int v88; // [rsp+8h] [rbp-48h]
  unsigned __int64 v89; // [rsp+10h] [rbp-40h]
  unsigned __int64 v90; // [rsp+18h] [rbp-38h]

  v5 = HIDWORD(a4);
  v6 = a2;
  v8 = a3;
  v9 = a4;
  v90 = HIDWORD(a2);
  v89 = HIDWORD(a3);
  result = 0;
  if ( (_DWORD)a1 )
  {
    if ( (_DWORD)a1 == 1 )
    {
      if ( HIDWORD(a1) <= 0x41 )
        goto LABEL_4;
      v56 = sub_CB72A0();
      v57 = sub_904010((__int64)v56, "minor Version (");
      v58 = sub_CB59D0(v57, HIDWORD(a1));
      v59 = sub_904010(v58, ") newer than tool ");
      v27 = 65;
      v24 = sub_904010(v59, "(should be ");
    }
    else
    {
      v88 = a1;
      v18 = sub_CB72A0();
      v19 = a1;
      v20 = v18[4];
      v21 = (__int64)v18;
      if ( (unsigned __int64)(v18[3] - v20) <= 8 )
      {
        v78 = sub_CB6200((__int64)v18, (unsigned __int8 *)"Version (", 9u);
        v19 = v88;
        v21 = v78;
      }
      else
      {
        *(_BYTE *)(v20 + 8) = 40;
        *(_QWORD *)v20 = 0x206E6F6973726556LL;
        v18[4] += 9LL;
      }
      v22 = sub_CB59D0(v21, v19);
      v23 = *(__m128i **)(v22 + 32);
      v24 = v22;
      if ( *(_QWORD *)(v22 + 24) - (_QWORD)v23 <= 0x10u )
      {
        v80 = sub_CB6200(v22, ") not compatible ", 0x11u);
        v26 = *(void **)(v80 + 32);
        v24 = v80;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
        v23[1].m128i_i8[0] = 32;
        *v23 = si128;
        v26 = (void *)(*(_QWORD *)(v22 + 32) + 17LL);
        *(_QWORD *)(v22 + 32) = v26;
      }
      if ( *(_QWORD *)(v24 + 24) - (_QWORD)v26 <= 0xAu )
      {
        v24 = sub_CB6200(v24, "(should be ", 0xBu);
      }
      else
      {
        qmemcpy(v26, "(should be ", 11);
        *(_QWORD *)(v24 + 32) += 11LL;
      }
      v27 = 1;
    }
    v28 = sub_CB59F0(v24, v27);
    sub_904010(v28, ")\n");
    result = 1;
  }
LABEL_4:
  if ( !v6 )
    goto LABEL_7;
  if ( v6 == 2 )
  {
    if ( (unsigned int)v90 <= 0x62 )
      goto LABEL_7;
    v60 = sub_CB72A0();
    v61 = sub_904010((__int64)v60, "minor NvvmIRVersion (");
    v62 = sub_CB59D0(v61, (unsigned int)v90);
    v63 = sub_904010(v62, ") newer than tool ");
    v64 = sub_904010(v63, "(should be ");
    v65 = sub_CB59F0(v64, 98);
    sub_904010(v65, ")\n");
  }
  else
  {
    v40 = sub_CB72A0();
    v41 = (void *)v40[4];
    v42 = (__int64)v40;
    if ( v40[3] - (_QWORD)v41 <= 0xEu )
    {
      v42 = sub_CB6200((__int64)v40, (unsigned __int8 *)"NvvmIRVersion (", 0xFu);
    }
    else
    {
      qmemcpy(v41, "NvvmIRVersion (", 15);
      v40[4] += 15LL;
    }
    v43 = sub_CB59D0(v42, v6);
    v44 = *(__m128i **)(v43 + 32);
    v45 = v43;
    if ( *(_QWORD *)(v43 + 24) - (_QWORD)v44 <= 0x10u )
    {
      v81 = sub_CB6200(v43, ") not compatible ", 0x11u);
      v47 = *(void **)(v81 + 32);
      v45 = v81;
    }
    else
    {
      v46 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v44[1].m128i_i8[0] = 32;
      *v44 = v46;
      v47 = (void *)(*(_QWORD *)(v43 + 32) + 17LL);
      *(_QWORD *)(v43 + 32) = v47;
    }
    if ( *(_QWORD *)(v45 + 24) - (_QWORD)v47 <= 0xAu )
    {
      v45 = sub_CB6200(v45, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v47, "(should be ", 11);
      *(_QWORD *)(v45 + 32) += 11LL;
    }
    v48 = sub_CB59F0(v45, 2);
    v49 = *(_WORD **)(v48 + 32);
    if ( *(_QWORD *)(v48 + 24) - (_QWORD)v49 <= 1u )
    {
      sub_CB6200(v48, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      *v49 = 2601;
      *(_QWORD *)(v48 + 32) += 2LL;
    }
  }
  result = 1;
LABEL_7:
  if ( !v8 )
    goto LABEL_10;
  if ( v8 == 3 )
  {
    if ( (unsigned int)v89 <= 2 )
      goto LABEL_10;
    v66 = sub_CB72A0();
    v67 = sub_904010((__int64)v66, "minor NvvmDebugVersion (");
    v68 = sub_CB59D0(v67, (unsigned int)v89);
    v69 = sub_904010(v68, ") newer than tool ");
    v70 = sub_904010(v69, "(should be ");
    v71 = sub_CB59F0(v70, 2);
    sub_904010(v71, ")\n");
  }
  else
  {
    v29 = sub_CB72A0();
    v30 = (__m128i *)v29[4];
    v31 = (__int64)v29;
    if ( v29[3] - (_QWORD)v30 <= 0x11u )
    {
      v31 = sub_CB6200((__int64)v29, (unsigned __int8 *)"NvvmDebugVersion (", 0x12u);
    }
    else
    {
      v32 = _mm_load_si128((const __m128i *)&xmmword_3F6E0F0);
      v30[1].m128i_i16[0] = 10272;
      *v30 = v32;
      v29[4] += 18LL;
    }
    v33 = sub_CB59D0(v31, v8);
    v34 = *(__m128i **)(v33 + 32);
    v35 = v33;
    if ( *(_QWORD *)(v33 + 24) - (_QWORD)v34 <= 0x10u )
    {
      v79 = sub_CB6200(v33, ") not compatible ", 0x11u);
      v37 = *(void **)(v79 + 32);
      v35 = v79;
    }
    else
    {
      v36 = _mm_load_si128((const __m128i *)&xmmword_3F6E0E0);
      v34[1].m128i_i8[0] = 32;
      *v34 = v36;
      v37 = (void *)(*(_QWORD *)(v33 + 32) + 17LL);
      *(_QWORD *)(v33 + 32) = v37;
    }
    if ( *(_QWORD *)(v35 + 24) - (_QWORD)v37 <= 0xAu )
    {
      v35 = sub_CB6200(v35, "(should be ", 0xBu);
    }
    else
    {
      qmemcpy(v37, "(should be ", 11);
      *(_QWORD *)(v35 + 32) += 11LL;
    }
    v38 = sub_CB59F0(v35, 3);
    v39 = *(_WORD **)(v38 + 32);
    if ( *(_QWORD *)(v38 + 24) - (_QWORD)v39 <= 1u )
    {
      sub_CB6200(v38, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      *v39 = 2601;
      *(_QWORD *)(v38 + 32) += 2LL;
    }
  }
  result = 1;
LABEL_10:
  if ( v9 )
  {
    if ( (_BYTE)a5 )
    {
      if ( v9 > 0x14 )
      {
        v72 = sub_CB72A0();
        v73 = sub_904010((__int64)v72, "LlvmVersion (");
        v74 = sub_CB59D0(v73, v9);
        v75 = sub_904010(v74, ") not compatible ");
        v76 = sub_904010(v75, "(should be ");
        v77 = sub_CB59F0(v76, 20);
        sub_904010(v77, ")\n");
        return a5;
      }
      else if ( (_DWORD)v5 != 0 && v9 == 20 )
      {
        v82 = sub_CB72A0();
        v83 = sub_904010((__int64)v82, "minor LlvmVersion (");
        v84 = sub_CB59D0(v83, (unsigned int)v5);
        v85 = sub_904010(v84, ") newer than tool ");
        v86 = sub_904010(v85, "(should be ");
        v87 = sub_CB59F0(v86, 0);
        sub_904010(v87, ")\n");
        return ((_DWORD)v5 != 0) & (unsigned __int8)(v9 == 20);
      }
    }
    else if ( v9 == 20 )
    {
      if ( (_DWORD)v5 )
      {
        v11 = sub_CB72A0();
        v12 = sub_904010((__int64)v11, "minor LlvmVersion (");
        v13 = sub_CB59D0(v12, (unsigned int)v5);
        v14 = sub_904010(v13, ") newer than tool ");
        v15 = sub_904010(v14, "(should be ");
        v16 = sub_CB59F0(v15, 0);
        v17 = sub_904010(v16, ").");
        sub_904010(v17, " Must be same as tool for ascii dumps\n");
        return 1;
      }
    }
    else
    {
      v50 = sub_CB72A0();
      v51 = sub_904010((__int64)v50, "LlvmVersion (");
      v52 = sub_CB59D0(v51, v9);
      v53 = sub_904010(v52, ") not compatible ");
      v54 = sub_904010(v53, "(should be ");
      v55 = sub_CB59F0(v54, 20);
      sub_904010(v55, ")\n");
      return 1;
    }
  }
  return result;
}
