// Function: sub_13A7B70
// Address: 0x13a7b70
//
__int64 __fastcall sub_13A7B70(__int64 a1, __m128i *a2, const __m128i *a3)
{
  __int32 v4; // eax
  __int32 v5; // edx
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned int v16; // r9d
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rbx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // r15
  char v34; // al
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rbx
  __int64 v58; // r11
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // r10
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // r14
  __int64 v67; // r14
  __int64 v68; // rax
  __int64 v69; // [rsp+0h] [rbp-F0h]
  __int64 v70; // [rsp+0h] [rbp-F0h]
  __int64 v71; // [rsp+8h] [rbp-E8h]
  __int64 v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 v74; // [rsp+18h] [rbp-D8h]
  __int64 v75; // [rsp+18h] [rbp-D8h]
  __int64 v76; // [rsp+20h] [rbp-D0h]
  unsigned __int8 v77; // [rsp+20h] [rbp-D0h]
  unsigned __int8 v78; // [rsp+2Fh] [rbp-C1h]
  __int64 v79; // [rsp+38h] [rbp-B8h]
  __int64 v80; // [rsp+38h] [rbp-B8h]
  __int64 v81; // [rsp+38h] [rbp-B8h]
  __int64 v82; // [rsp+38h] [rbp-B8h]
  __int64 v83; // [rsp+38h] [rbp-B8h]
  __int64 v84; // [rsp+38h] [rbp-B8h]
  __int64 v85; // [rsp+38h] [rbp-B8h]
  __int64 v86; // [rsp+38h] [rbp-B8h]
  __int64 v87; // [rsp+38h] [rbp-B8h]
  __int64 v88[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v89[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v90[2]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v91[2]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v92[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v93[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v94[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v95[8]; // [rsp+B0h] [rbp-40h] BYREF

  v4 = a2->m128i_i32[0];
  if ( a2->m128i_i32[0] == 4 )
  {
    if ( a3->m128i_i32[0] == 4 )
      return 0;
    goto LABEL_11;
  }
  if ( !v4 )
    return 0;
  v5 = a3->m128i_i32[0];
  if ( !v5 )
  {
LABEL_14:
    sub_13A6370(a2);
    return 1;
  }
  if ( v4 == 2 && v5 == 2 )
  {
    v35 = sub_13A62A0((__int64)a3);
    v36 = sub_13A62A0((__int64)a2);
    if ( (unsigned __int8)sub_13A7760(a1, 32, v36, v35) )
      return 0;
    v37 = sub_13A62A0((__int64)a3);
    v38 = sub_13A62A0((__int64)a2);
    if ( (unsigned __int8)sub_13A7760(a1, 33, v38, v37) )
      goto LABEL_14;
    if ( *(_WORD *)(sub_13A62A0((__int64)a3) + 24) )
      return 0;
LABEL_11:
    v16 = 1;
    *a2 = _mm_loadu_si128(a3);
    a2[1] = _mm_loadu_si128(a3 + 1);
    a2[2] = _mm_loadu_si128(a3 + 2);
    return v16;
  }
  if ( (unsigned int)(v4 - 2) > 1 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = sub_13A6250((__int64)a2);
    v9 = sub_13A6270((__int64)a3);
    v10 = sub_13A5B60(v7, v9, v8, 0, 0);
    v79 = *(_QWORD *)(a1 + 8);
    v11 = sub_13A6260((__int64)a2);
    v12 = sub_13A6280((__int64)a3);
    v13 = sub_13A5B60(v79, v12, v11, 0, 0);
    v14 = sub_13A5B00(*(_QWORD *)(a1 + 8), v10, v13, 0, 0);
    v15 = sub_13A6290((__int64)a3);
    if ( (unsigned __int8)sub_13A7760(a1, 32, v14, v15) )
      return 0;
    v18 = sub_13A6290((__int64)a3);
    goto LABEL_13;
  }
  v19 = *(_QWORD *)(a1 + 8);
  v20 = sub_13A6280((__int64)a3);
  v21 = sub_13A6270((__int64)a2);
  v22 = sub_13A5B60(v19, v21, v20, 0, 0);
  v80 = *(_QWORD *)(a1 + 8);
  v23 = sub_13A6270((__int64)a3);
  v24 = sub_13A6280((__int64)a2);
  v25 = sub_13A5B60(v80, v24, v23, 0, 0);
  if ( (unsigned __int8)sub_13A7760(a1, 32, v22, v25) )
  {
    v26 = *(_QWORD *)(a1 + 8);
    v27 = sub_13A6280((__int64)a3);
    v28 = sub_13A6290((__int64)a2);
    v29 = sub_13A5B60(v26, v28, v27, 0, 0);
    v30 = *(_QWORD *)(a1 + 8);
    v14 = v29;
    v31 = sub_13A6290((__int64)a3);
    v32 = sub_13A6280((__int64)a2);
    v33 = sub_13A5B60(v30, v32, v31, 0, 0);
    v34 = sub_13A7760(a1, 32, v14, v33);
    v18 = v33;
    if ( v34 )
      return 0;
LABEL_13:
    if ( !(unsigned __int8)sub_13A7760(a1, 33, v14, v18) )
      return 0;
    goto LABEL_14;
  }
  v78 = sub_13A7760(a1, 33, v22, v25);
  if ( !v78 )
    return 0;
  v39 = *(_QWORD *)(a1 + 8);
  v81 = sub_13A6280((__int64)a3);
  v40 = sub_13A6290((__int64)a2);
  v41 = sub_13A5B60(v39, v40, v81, 0, 0);
  v42 = *(_QWORD *)(a1 + 8);
  v74 = v41;
  v82 = sub_13A6270((__int64)a3);
  v43 = sub_13A6290((__int64)a2);
  v44 = sub_13A5B60(v42, v43, v82, 0, 0);
  v45 = *(_QWORD *)(a1 + 8);
  v69 = v44;
  v83 = sub_13A6280((__int64)a2);
  v46 = sub_13A6290((__int64)a3);
  v47 = sub_13A5B60(v45, v46, v83, 0, 0);
  v48 = *(_QWORD *)(a1 + 8);
  v72 = v47;
  v84 = sub_13A6270((__int64)a2);
  v49 = sub_13A6290((__int64)a3);
  v50 = sub_13A5B60(v48, v49, v84, 0, 0);
  v51 = *(_QWORD *)(a1 + 8);
  v71 = v50;
  v85 = sub_13A6280((__int64)a3);
  v52 = sub_13A6270((__int64)a2);
  v53 = sub_13A5B60(v51, v52, v85, 0, 0);
  v54 = *(_QWORD *)(a1 + 8);
  v86 = v53;
  v76 = sub_13A6280((__int64)a2);
  v55 = sub_13A6270((__int64)a3);
  v56 = sub_13A5B60(v54, v55, v76, 0, 0);
  v77 = 0;
  v57 = sub_14806B0(*(_QWORD *)(a1 + 8), v69, v71, 0, 0);
  if ( *(_WORD *)(v57 + 24) )
  {
    v57 = 0;
    v77 = v78;
  }
  v58 = sub_14806B0(*(_QWORD *)(a1 + 8), v74, v72, 0, 0);
  if ( *(_WORD *)(v58 + 24) )
  {
    v58 = 0;
    v77 = v78;
  }
  v73 = v58;
  v59 = sub_14806B0(*(_QWORD *)(a1 + 8), v86, v56, 0, 0);
  v60 = *(_QWORD *)(a1 + 8);
  v61 = v59;
  if ( *(_WORD *)(v59 + 24) )
    v61 = 0;
  v75 = v61;
  v62 = sub_14806B0(v60, v56, v86, 0, 0);
  v16 = 0;
  v87 = v62;
  if ( !*(_WORD *)(v62 + 24) && !v77 )
  {
    v16 = 0;
    if ( v75 )
    {
      sub_13A38D0((__int64)v88, *(_QWORD *)(v73 + 32) + 24LL);
      sub_13A38D0((__int64)v89, *(_QWORD *)(v75 + 32) + 24LL);
      sub_13A38D0((__int64)v90, *(_QWORD *)(v57 + 32) + 24LL);
      sub_13A38D0((__int64)v91, *(_QWORD *)(v87 + 32) + 24LL);
      sub_13A38D0((__int64)v92, (__int64)v88);
      sub_13A38D0((__int64)v93, (__int64)v88);
      sub_16AE5C0(v88, v89, v92, v93);
      sub_13A38D0((__int64)v94, (__int64)v90);
      sub_13A38D0((__int64)v95, (__int64)v90);
      sub_16AE5C0(v90, v91, v94, v95);
      if ( !sub_13A38F0((__int64)v93, 0)
        || !sub_13A38F0((__int64)v95, 0)
        || sub_13A3940((__int64)v92, 0)
        || sub_13A3940((__int64)v94, 0)
        || (v63 = sub_1456040(v22), v64 = sub_13A62B0((__int64)a2), (v65 = sub_13A7B50(a1, v64, v63)) != 0)
        && ((v66 = *(_QWORD *)(v65 + 32) + 24LL, (int)sub_16AEA10(v92, v66) > 0) || (int)sub_16AEA10(v94, v66) > 0) )
      {
        sub_13A6370(a2);
      }
      else
      {
        v70 = sub_13A62B0((__int64)a2);
        v67 = sub_145CF40(*(_QWORD *)(a1 + 8), v94);
        v68 = sub_145CF40(*(_QWORD *)(a1 + 8), v92);
        sub_13A62C0((__int64)a2, v68, v67, v70);
      }
      sub_135E100(v95);
      sub_135E100(v94);
      sub_135E100(v93);
      sub_135E100(v92);
      sub_135E100(v91);
      sub_135E100(v90);
      sub_135E100(v89);
      sub_135E100(v88);
      return v78;
    }
  }
  return v16;
}
