// Function: sub_2C42830
// Address: 0x2c42830
//
__int64 __fastcall sub_2C42830(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rcx
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  _QWORD *v66; // r9
  _QWORD *v67; // r8
  __int64 v68; // rcx
  __int64 v69; // rsi
  __int64 v70; // r14
  char v72; // al
  char v73; // al
  _QWORD *v74; // [rsp+8h] [rbp-838h]
  _QWORD v75[15]; // [rsp+50h] [rbp-7F0h] BYREF
  __int16 v76; // [rsp+C8h] [rbp-778h]
  _QWORD v77[15]; // [rsp+D0h] [rbp-770h] BYREF
  __int16 v78; // [rsp+148h] [rbp-6F8h]
  _QWORD v79[15]; // [rsp+150h] [rbp-6F0h] BYREF
  __int16 v80; // [rsp+1C8h] [rbp-678h]
  _QWORD v81[15]; // [rsp+1D0h] [rbp-670h] BYREF
  __int16 v82; // [rsp+248h] [rbp-5F8h]
  _QWORD v83[15]; // [rsp+250h] [rbp-5F0h] BYREF
  __int16 v84; // [rsp+2C8h] [rbp-578h]
  _QWORD v85[15]; // [rsp+2D0h] [rbp-570h] BYREF
  __int16 v86; // [rsp+348h] [rbp-4F8h]
  _QWORD v87[15]; // [rsp+350h] [rbp-4F0h] BYREF
  __int16 v88; // [rsp+3C8h] [rbp-478h]
  _QWORD v89[4]; // [rsp+3D0h] [rbp-470h] BYREF
  char v90[64]; // [rsp+3F0h] [rbp-450h] BYREF
  __int64 v91; // [rsp+430h] [rbp-410h]
  __int64 v92; // [rsp+438h] [rbp-408h]
  __int64 v93; // [rsp+440h] [rbp-400h]
  __int16 v94; // [rsp+448h] [rbp-3F8h]
  _QWORD v95[12]; // [rsp+450h] [rbp-3F0h] BYREF
  __int64 v96; // [rsp+4B0h] [rbp-390h]
  __int64 v97; // [rsp+4B8h] [rbp-388h]
  __int16 v98; // [rsp+4C8h] [rbp-378h]
  _BYTE v99[32]; // [rsp+4E0h] [rbp-360h] BYREF
  char v100[64]; // [rsp+500h] [rbp-340h] BYREF
  __int64 v101; // [rsp+540h] [rbp-300h]
  __int64 v102; // [rsp+548h] [rbp-2F8h]
  __int64 v103; // [rsp+550h] [rbp-2F0h]
  __int16 v104; // [rsp+558h] [rbp-2E8h]
  _QWORD v105[12]; // [rsp+560h] [rbp-2E0h] BYREF
  __int64 v106; // [rsp+5C0h] [rbp-280h]
  __int64 v107; // [rsp+5C8h] [rbp-278h]
  __int16 v108; // [rsp+5D8h] [rbp-268h]
  _QWORD v109[15]; // [rsp+5F0h] [rbp-250h] BYREF
  __int16 v110; // [rsp+668h] [rbp-1D8h]
  char v111[144]; // [rsp+670h] [rbp-1D0h] BYREF
  _QWORD v112[4]; // [rsp+700h] [rbp-140h] BYREF
  _BYTE v113[64]; // [rsp+720h] [rbp-120h] BYREF
  __int64 v114; // [rsp+760h] [rbp-E0h]
  __int64 v115; // [rsp+768h] [rbp-D8h]
  __int64 v116; // [rsp+770h] [rbp-D0h]
  __int16 v117; // [rsp+778h] [rbp-C8h]
  char v118[192]; // [rsp+780h] [rbp-C0h] BYREF

  sub_2ABD910(v75, a2, a3, a4, a5, a6);
  v76 = *(_WORD *)(a2 + 120);
  sub_2ABD910(v77, a2 + 128, v6, v7, v8, v9);
  v78 = *(_WORD *)(a2 + 248);
  sub_2ABD910(v85, (__int64)v77, v10, v11, v12, v13);
  v86 = v78;
  sub_2ABD910(v83, (__int64)v77, v14, v15, v16, v17);
  v84 = v78;
  sub_2ABD910(v109, (__int64)v85, v18, v19, v20, v21);
  v110 = v86;
  sub_2ABD910(v89, (__int64)v83, v22, v23, v24, v25);
  v94 = v84;
  sub_2ABD910(v112, (__int64)v89, v26, v27, v28, v29);
  v117 = v94;
  sub_C8CF70((__int64)v99, v100, 8, (__int64)v113, (__int64)v112);
  v30 = v114;
  v114 = 0;
  v101 = v30;
  v31 = v115;
  v115 = 0;
  v102 = v31;
  v32 = v116;
  v116 = 0;
  v103 = v32;
  v104 = v117;
  sub_2AB1B10((__int64)v112);
  sub_2ABD910(v105, (__int64)v109, v33, v34, v35, v36);
  v108 = v110;
  while ( 1 )
  {
    v39 = v101;
    v40 = v106;
    if ( v102 - v101 != v107 - v106 )
      goto LABEL_3;
    if ( v101 == v102 )
      break;
    while ( *(_QWORD *)v39 == *(_QWORD *)v40 )
    {
      v72 = *(_BYTE *)(v39 + 16);
      if ( v72 != *(_BYTE *)(v40 + 16) || v72 && *(_QWORD *)(v39 + 8) != *(_QWORD *)(v40 + 8) )
        break;
      v39 += 24;
      v40 += 24;
      if ( v102 == v39 )
        goto LABEL_4;
    }
LABEL_3:
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v102 - 24) + 8LL) - 1 <= 1 )
      break;
    sub_2ADA290((__int64)v99, v40, v102, v39, v37, v38);
  }
LABEL_4:
  sub_2AB1B10((__int64)v89);
  sub_2AB1B10((__int64)v109);
  sub_2ABD910(v81, (__int64)v77, v41, v42, v43, v44);
  v82 = v78;
  sub_2ABD910(v79, (__int64)v75, v45, v46, v47, v48);
  v80 = v76;
  sub_2ABD910(v109, (__int64)v81, v49, v50, v51, v52);
  v110 = v82;
  sub_2ABD910(v87, (__int64)v79, v53, v54, v55, (__int64)v87);
  v88 = v80;
  sub_2ABD910(v112, (__int64)v87, v56, v57, v58, (__int64)v87);
  v117 = v88;
  sub_C8CF70((__int64)v89, v90, 8, (__int64)v113, (__int64)v112);
  v59 = v114;
  v114 = 0;
  v91 = v59;
  v60 = v115;
  v115 = 0;
  v92 = v60;
  v61 = v116;
  v116 = 0;
  v93 = v61;
  v94 = v117;
  sub_2AB1B10((__int64)v112);
  sub_2ABD910(v95, (__int64)v109, v62, v63, v64, v65);
  v66 = v87;
  v67 = v89;
  v98 = v110;
  while ( 1 )
  {
    v68 = v91;
    v69 = v96;
    if ( v92 - v91 != v97 - v96 )
      goto LABEL_6;
    if ( v91 == v92 )
      break;
    while ( *(_QWORD *)v68 == *(_QWORD *)v69 )
    {
      v73 = *(_BYTE *)(v68 + 16);
      if ( v73 != *(_BYTE *)(v69 + 16) || v73 && *(_QWORD *)(v68 + 8) != *(_QWORD *)(v69 + 8) )
        break;
      v68 += 24;
      v69 += 24;
      if ( v92 == v68 )
        goto LABEL_7;
    }
LABEL_6:
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v92 - 24) + 8LL) - 1 <= 1 )
      break;
    v74 = v67;
    sub_2ADA290((__int64)v67, v69, v92, v68, (__int64)v67, (__int64)v66);
    v67 = v74;
  }
LABEL_7:
  v70 = (__int64)v67;
  sub_2AB1B10((__int64)v87);
  sub_2AB1B10((__int64)v109);
  sub_2C2BB00((__int64)v112, (__int64)v99);
  sub_2C2BB00((__int64)v109, v70);
  sub_2C2BB00(a1, (__int64)v109);
  sub_2C2BB00(a1 + 264, (__int64)v112);
  sub_2AB1B10((__int64)v111);
  sub_2AB1B10((__int64)v109);
  sub_2AB1B10((__int64)v118);
  sub_2AB1B10((__int64)v112);
  sub_2AB1B10((__int64)v95);
  sub_2AB1B10(v70);
  sub_2AB1B10((__int64)v79);
  sub_2AB1B10((__int64)v81);
  sub_2AB1B10((__int64)v105);
  sub_2AB1B10((__int64)v99);
  sub_2AB1B10((__int64)v83);
  sub_2AB1B10((__int64)v85);
  sub_2AB1B10((__int64)v77);
  sub_2AB1B10((__int64)v75);
  return a1;
}
