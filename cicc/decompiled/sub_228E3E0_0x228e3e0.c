// Function: sub_228E3E0
// Address: 0x228e3e0
//
__int64 __fastcall sub_228E3E0(__int64 a1, __m128i *a2, const __m128i *a3)
{
  __int32 v4; // eax
  __int32 v5; // edx
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // rdi
  _QWORD *v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r9d
  __int64 v19; // rcx
  __int64 *v20; // rbx
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 *v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 *v34; // r15
  char v35; // al
  __int64 *v36; // r14
  __int64 *v37; // rax
  __int64 *v38; // r14
  __int64 *v39; // rax
  __int64 *v40; // rbx
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rbx
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rbx
  __int64 v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rbx
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rbx
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 *v55; // rbx
  __int64 v56; // rax
  __int64 *v57; // r15
  _QWORD *v58; // rbx
  _QWORD *v59; // r11
  _QWORD *v60; // rax
  __int64 *v61; // rdi
  _QWORD *v62; // r10
  _QWORD *v63; // rax
  __int64 v64; // r14
  char *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // r14
  _QWORD *v68; // r14
  _QWORD *v69; // rax
  __int64 v70; // [rsp+0h] [rbp-100h]
  __int64 v71; // [rsp+0h] [rbp-100h]
  __int64 v72; // [rsp+8h] [rbp-F8h]
  __int64 v73; // [rsp+10h] [rbp-F0h]
  _QWORD *v74; // [rsp+10h] [rbp-F0h]
  __int64 v75; // [rsp+18h] [rbp-E8h]
  _QWORD *v76; // [rsp+18h] [rbp-E8h]
  __int64 v77; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v78; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v79; // [rsp+2Fh] [rbp-D1h]
  __int64 *v80; // [rsp+38h] [rbp-C8h]
  __int64 *v81; // [rsp+38h] [rbp-C8h]
  __int64 v82; // [rsp+38h] [rbp-C8h]
  __int64 v83; // [rsp+38h] [rbp-C8h]
  __int64 v84; // [rsp+38h] [rbp-C8h]
  __int64 v85; // [rsp+38h] [rbp-C8h]
  __int64 v86; // [rsp+38h] [rbp-C8h]
  __int64 v87; // [rsp+38h] [rbp-C8h]
  _QWORD *v88; // [rsp+38h] [rbp-C8h]
  __int64 v89[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v90[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v91[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v92[2]; // [rsp+70h] [rbp-90h] BYREF
  __int64 v93[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v94[2]; // [rsp+90h] [rbp-70h] BYREF
  __int64 v95[2]; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v96[2]; // [rsp+B0h] [rbp-50h] BYREF
  _QWORD v97[8]; // [rsp+C0h] [rbp-40h] BYREF

  v4 = a2->m128i_i32[0];
  if ( a2->m128i_i32[0] == 4 )
  {
    if ( a3->m128i_i32[0] == 4 )
      return 0;
    goto LABEL_15;
  }
  if ( !v4 )
    return 0;
  v5 = a3->m128i_i32[0];
  if ( !v5 )
  {
LABEL_18:
    sub_228CEE0(a2);
    return 1;
  }
  if ( v5 == 2 && v4 == 2 )
  {
    v36 = sub_228CE10((__int64)a3);
    v37 = sub_228CE10((__int64)a2);
    if ( sub_228DFC0(a1, 0x20u, (__int64)v37, (__int64)v36) )
      return 0;
    v38 = sub_228CE10((__int64)a3);
    v39 = sub_228CE10((__int64)a2);
    if ( sub_228DFC0(a1, 0x21u, (__int64)v39, (__int64)v38) )
      goto LABEL_18;
    if ( *((_WORD *)sub_228CE10((__int64)a3) + 12) )
      return 0;
LABEL_15:
    v17 = 1;
    *a2 = _mm_loadu_si128(a3);
    a2[1] = _mm_loadu_si128(a3 + 1);
    a2[2] = _mm_loadu_si128(a3 + 2);
    return v17;
  }
  if ( (unsigned int)(v4 - 2) > 1 )
  {
    if ( v4 == 1 && (unsigned int)(v5 - 2) <= 1 )
    {
      v7 = *(__int64 **)(a1 + 8);
      v8 = sub_228CDC0((__int64)a2);
      v9 = sub_228CDE0((__int64)a3);
      v10 = sub_DCA690(v7, v9, v8, 0, 0);
      v80 = *(__int64 **)(a1 + 8);
      v11 = sub_228CDD0((__int64)a2);
      v12 = sub_228CDF0((__int64)a3);
      v13 = sub_DCA690(v80, v12, v11, 0, 0);
      v14 = *(__int64 **)(a1 + 8);
      v97[1] = v13;
      v96[0] = (__int64)v97;
      v97[0] = v10;
      v96[1] = 0x200000002LL;
      v15 = sub_DC7EB0(v14, (__int64)v96, 0, 0);
      if ( (_QWORD *)v96[0] != v97 )
        _libc_free(v96[0]);
      v16 = sub_228CE00((__int64)a3);
      if ( sub_228DFC0(a1, 0x20u, (__int64)v15, v16) )
        return 0;
      v19 = sub_228CE00((__int64)a3);
LABEL_17:
      if ( sub_228DFC0(a1, 0x21u, (__int64)v15, v19) )
        goto LABEL_18;
      return 0;
    }
LABEL_47:
    BUG();
  }
  if ( (unsigned int)(v5 - 2) > 1 )
    goto LABEL_47;
  v20 = *(__int64 **)(a1 + 8);
  v21 = sub_228CDF0((__int64)a3);
  v22 = sub_228CDE0((__int64)a2);
  v23 = sub_DCA690(v20, v22, v21, 0, 0);
  v81 = *(__int64 **)(a1 + 8);
  v24 = sub_228CDE0((__int64)a3);
  v25 = sub_228CDF0((__int64)a2);
  v26 = sub_DCA690(v81, v25, v24, 0, 0);
  if ( sub_228DFC0(a1, 0x20u, (__int64)v23, (__int64)v26) )
  {
    v27 = *(__int64 **)(a1 + 8);
    v28 = sub_228CDF0((__int64)a3);
    v29 = sub_228CE00((__int64)a2);
    v30 = sub_DCA690(v27, v29, v28, 0, 0);
    v31 = *(__int64 **)(a1 + 8);
    v15 = v30;
    v32 = sub_228CE00((__int64)a3);
    v33 = sub_228CDF0((__int64)a2);
    v34 = sub_DCA690(v31, v33, v32, 0, 0);
    v35 = sub_228DFC0(a1, 0x20u, (__int64)v15, (__int64)v34);
    v19 = (__int64)v34;
    if ( v35 )
      return 0;
    goto LABEL_17;
  }
  v79 = sub_228DFC0(a1, 0x21u, (__int64)v23, (__int64)v26);
  if ( !v79 )
    return 0;
  v40 = *(__int64 **)(a1 + 8);
  v82 = sub_228CDF0((__int64)a3);
  v41 = sub_228CE00((__int64)a2);
  v42 = sub_DCA690(v40, v41, v82, 0, 0);
  v43 = *(__int64 **)(a1 + 8);
  v75 = (__int64)v42;
  v83 = sub_228CDE0((__int64)a3);
  v44 = sub_228CE00((__int64)a2);
  v45 = sub_DCA690(v43, v44, v83, 0, 0);
  v46 = *(__int64 **)(a1 + 8);
  v70 = (__int64)v45;
  v84 = sub_228CDF0((__int64)a2);
  v47 = sub_228CE00((__int64)a3);
  v48 = sub_DCA690(v46, v47, v84, 0, 0);
  v49 = *(__int64 **)(a1 + 8);
  v73 = (__int64)v48;
  v85 = sub_228CDE0((__int64)a2);
  v50 = sub_228CE00((__int64)a3);
  v51 = sub_DCA690(v49, v50, v85, 0, 0);
  v52 = *(__int64 **)(a1 + 8);
  v72 = (__int64)v51;
  v86 = sub_228CDF0((__int64)a3);
  v53 = sub_228CDE0((__int64)a2);
  v54 = sub_DCA690(v52, v53, v86, 0, 0);
  v55 = *(__int64 **)(a1 + 8);
  v87 = (__int64)v54;
  v77 = sub_228CDF0((__int64)a2);
  v56 = sub_228CDE0((__int64)a3);
  v57 = sub_DCA690(v55, v56, v77, 0, 0);
  v78 = 0;
  v58 = sub_DCC810(*(__int64 **)(a1 + 8), v70, v72, 0, 0);
  if ( *((_WORD *)v58 + 12) )
  {
    v58 = 0;
    v78 = v79;
  }
  v59 = sub_DCC810(*(__int64 **)(a1 + 8), v75, v73, 0, 0);
  if ( *((_WORD *)v59 + 12) )
  {
    v59 = 0;
    v78 = v79;
  }
  v74 = v59;
  v60 = sub_DCC810(*(__int64 **)(a1 + 8), v87, (__int64)v57, 0, 0);
  v61 = *(__int64 **)(a1 + 8);
  v62 = v60;
  if ( *((_WORD *)v60 + 12) )
    v62 = 0;
  v76 = v62;
  v63 = sub_DCC810(v61, (__int64)v57, v87, 0, 0);
  v17 = 0;
  v88 = v63;
  if ( !*((_WORD *)v63 + 12) && !v78 )
  {
    v17 = 0;
    if ( v76 )
    {
      sub_9865C0((__int64)v89, v74[4] + 24LL);
      sub_9865C0((__int64)v90, v76[4] + 24LL);
      sub_9865C0((__int64)v91, v58[4] + 24LL);
      sub_9865C0((__int64)v92, v88[4] + 24LL);
      sub_9865C0((__int64)v93, (__int64)v89);
      sub_9865C0((__int64)v94, (__int64)v89);
      sub_C4C400((__int64)v89, (__int64)v90, (__int64)v93, (__int64)v94);
      sub_9865C0((__int64)v95, (__int64)v91);
      sub_9865C0((__int64)v96, (__int64)v91);
      sub_C4C400((__int64)v91, (__int64)v92, (__int64)v95, (__int64)v96);
      if ( !sub_D94970((__int64)v94, 0)
        || !sub_D94970((__int64)v96, 0)
        || sub_986F30((__int64)v93, 0)
        || sub_986F30((__int64)v95, 0)
        || (v64 = sub_D95540((__int64)v23),
            v65 = (char *)sub_228CE20((__int64)a2),
            (v66 = sub_228E3C0(a1, v65, v64)) != 0)
        && ((v67 = v66[4] + 24LL, (int)sub_C4C880((__int64)v93, v67) > 0) || (int)sub_C4C880((__int64)v95, v67) > 0) )
      {
        sub_228CEE0(a2);
      }
      else
      {
        v71 = sub_228CE20((__int64)a2);
        v68 = sub_DA26C0(*(__int64 **)(a1 + 8), (__int64)v95);
        v69 = sub_DA26C0(*(__int64 **)(a1 + 8), (__int64)v93);
        sub_228CE30((__int64)a2, (__int64)v69, (__int64)v68, v71);
      }
      sub_969240(v96);
      sub_969240(v95);
      sub_969240(v94);
      sub_969240(v93);
      sub_969240(v92);
      sub_969240(v91);
      sub_969240(v90);
      sub_969240(v89);
      return v79;
    }
  }
  return v17;
}
