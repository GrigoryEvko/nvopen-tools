// Function: sub_2C31060
// Address: 0x2c31060
//
__int64 __fastcall sub_2C31060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  _QWORD v80[15]; // [rsp+80h] [rbp-BA0h] BYREF
  __int16 v81; // [rsp+F8h] [rbp-B28h]
  _QWORD v82[15]; // [rsp+100h] [rbp-B20h] BYREF
  __int16 v83; // [rsp+178h] [rbp-AA8h]
  _QWORD v84[15]; // [rsp+180h] [rbp-AA0h] BYREF
  unsigned __int16 v85; // [rsp+1F8h] [rbp-A28h]
  _QWORD v86[15]; // [rsp+200h] [rbp-A20h] BYREF
  __int16 v87; // [rsp+278h] [rbp-9A8h]
  _QWORD v88[15]; // [rsp+280h] [rbp-9A0h] BYREF
  unsigned __int16 v89; // [rsp+2F8h] [rbp-928h]
  _QWORD v90[15]; // [rsp+300h] [rbp-920h] BYREF
  unsigned __int16 v91; // [rsp+378h] [rbp-8A8h]
  _QWORD v92[15]; // [rsp+380h] [rbp-8A0h] BYREF
  __int16 v93; // [rsp+3F8h] [rbp-828h]
  _QWORD v94[15]; // [rsp+400h] [rbp-820h] BYREF
  unsigned __int16 v95; // [rsp+478h] [rbp-7A8h]
  _QWORD v96[15]; // [rsp+490h] [rbp-790h] BYREF
  __int16 v97; // [rsp+508h] [rbp-718h]
  _QWORD v98[15]; // [rsp+510h] [rbp-710h] BYREF
  unsigned __int16 v99; // [rsp+588h] [rbp-698h]
  _QWORD v100[15]; // [rsp+5A0h] [rbp-680h] BYREF
  unsigned __int16 v101; // [rsp+618h] [rbp-608h]
  _QWORD v102[15]; // [rsp+620h] [rbp-600h] BYREF
  unsigned __int16 v103; // [rsp+698h] [rbp-588h]
  __int16 v104; // [rsp+6A8h] [rbp-578h]
  _QWORD v105[15]; // [rsp+6B0h] [rbp-570h] BYREF
  __int16 v106; // [rsp+728h] [rbp-4F8h]
  _QWORD v107[15]; // [rsp+730h] [rbp-4F0h] BYREF
  unsigned __int16 v108; // [rsp+7A8h] [rbp-478h]
  __int16 v109; // [rsp+7B8h] [rbp-468h]
  _QWORD v110[15]; // [rsp+7C0h] [rbp-460h] BYREF
  __int16 v111; // [rsp+838h] [rbp-3E8h]
  _QWORD v112[15]; // [rsp+840h] [rbp-3E0h] BYREF
  unsigned __int16 v113; // [rsp+8B8h] [rbp-368h]
  __int16 v114; // [rsp+8C8h] [rbp-358h]
  _QWORD v115[15]; // [rsp+8D0h] [rbp-350h] BYREF
  unsigned __int16 v116; // [rsp+948h] [rbp-2D8h]
  _QWORD v117[15]; // [rsp+950h] [rbp-2D0h] BYREF
  unsigned __int16 v118; // [rsp+9C8h] [rbp-258h]
  __int16 v119; // [rsp+9D8h] [rbp-248h]
  _QWORD v120[15]; // [rsp+9E0h] [rbp-240h] BYREF
  unsigned __int16 v121; // [rsp+A58h] [rbp-1C8h]
  _QWORD v122[15]; // [rsp+A60h] [rbp-1C0h] BYREF
  unsigned __int16 v123; // [rsp+AD8h] [rbp-148h]
  _QWORD v124[15]; // [rsp+AE8h] [rbp-138h] BYREF
  unsigned __int16 v125; // [rsp+B60h] [rbp-C0h]
  _QWORD v126[15]; // [rsp+B68h] [rbp-B8h] BYREF
  unsigned __int16 v127; // [rsp+BE0h] [rbp-40h]

  sub_2ABCC20(v96, a2 + 120, a3, a4, a5, a6);
  sub_2C2B3B0(v115, v96);
  sub_2C2B3B0(v120, v115);
  sub_2C2B3B0(v105, v120);
  sub_2AB1B50((__int64)v120);
  v106 = 256;
  sub_2AB1B50((__int64)v115);
  sub_2ABCC20(v100, a2, v6, v7, v8, v9);
  sub_2C2B3B0(v115, v100);
  sub_2C2B3B0(v120, v115);
  sub_2C2B3B0(v110, v120);
  sub_2AB1B50((__int64)v120);
  v111 = 256;
  sub_2AB1B50((__int64)v115);
  sub_2C2B3B0(v120, v105);
  v121 = v106;
  sub_2C2B3B0(v115, v110);
  v116 = v111;
  sub_2C2B3B0(v88, v115);
  v89 = v116;
  sub_2C2B3B0(v90, v120);
  v91 = v121;
  sub_2AB1B50((__int64)v115);
  sub_2AB1B50((__int64)v120);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v100);
  sub_2AB1B50((__int64)v105);
  sub_2AB1B50((__int64)v96);
  sub_2ABCC20(v80, (__int64)v88, v10, v11, (__int64)v88, v12);
  v81 = v89;
  sub_2ABCC20(v82, (__int64)v90, v89, v13, v14, (__int64)v90);
  v83 = v91;
  sub_2ABCC20(v84, (__int64)v82, (__int64)v84, v15, v16, v17);
  v85 = v83;
  sub_2ABCC20(v86, (__int64)v82, v18, (__int64)v86, v19, v20);
  v87 = v83;
  sub_2ABCC20(v115, (__int64)v84, (__int64)v84, v21, v22, v23);
  v116 = v85;
  sub_2ABCC20(v110, (__int64)v86, v24, (__int64)v86, v25, v26);
  v111 = v87;
  sub_2ABCC20(v120, (__int64)v110, v27, v28, v29, v30);
  v121 = v111;
  sub_2C2B3B0(v100, v120);
  v101 = v121;
  sub_2AB1B50((__int64)v120);
  sub_2ABCC20(v102, (__int64)v115, v31, v32, v33, v34);
  v103 = v116;
  sub_2C30FC0(v100, v116, v35, v36, v37, v38);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v115);
  sub_2ABCC20(v92, (__int64)v82, v39, (__int64)v92, v40, v41);
  v93 = v83;
  sub_2ABCC20(v96, (__int64)v80, v42, v43, v44, v45);
  v97 = v81;
  sub_2ABCC20(v115, (__int64)v92, v46, v47, v48, v49);
  v116 = v93;
  sub_2ABCC20(v110, (__int64)v96, v50, v51, v52, v53);
  v111 = v97;
  sub_2ABCC20(v120, (__int64)v110, v54, v55, v56, v57);
  v121 = v111;
  sub_2C2B3B0(v105, v120);
  v106 = v121;
  sub_2AB1B50((__int64)v120);
  sub_2ABCC20(v107, (__int64)v115, (__int64)v107, v58, v59, v60);
  v108 = v116;
  sub_2C30FC0(v105, v116, v61, v62, v63, v64);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v115);
  sub_2C2B3B0(v115, v100);
  v116 = v101;
  sub_2C2B3B0(v117, v102);
  v118 = v103;
  sub_2C2B3B0(v110, v105);
  v111 = v106;
  sub_2C2B3B0(v112, v107);
  v113 = v108;
  sub_2C2B3B0(v120, v110);
  v121 = v111;
  sub_2C2B3B0(v122, v112);
  v123 = v113;
  sub_2C2B3B0(v124, v115);
  v125 = v116;
  sub_2C2B3B0(v126, v117);
  v127 = v118;
  sub_2AB1B50((__int64)v112);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v117);
  sub_2AB1B50((__int64)v115);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v105);
  sub_2AB1B50((__int64)v96);
  sub_2AB1B50((__int64)v92);
  sub_2AB1B50((__int64)v102);
  sub_2AB1B50((__int64)v100);
  sub_2AB1B50((__int64)v86);
  sub_2AB1B50((__int64)v84);
  sub_2AB1B50((__int64)v82);
  sub_2AB1B50((__int64)v80);
  sub_2ABCC20(v92, (__int64)v124, v65, v66, v67, v68);
  v93 = v125;
  sub_2ABCC20(v94, (__int64)v126, v125, v69, v70, v71);
  v95 = v127;
  sub_2C2B3B0(v110, v92);
  v111 = v93;
  sub_2C2B3B0(v112, v94);
  v113 = v95;
  sub_2C2B3B0(v115, v110);
  v116 = v111;
  sub_2C2B3B0(v117, v112);
  v118 = v113;
  sub_2C2B3B0(v100, v115);
  v101 = v116;
  sub_2C2B3B0(v102, v117);
  v103 = v118;
  sub_2AB1B50((__int64)v117);
  sub_2AB1B50((__int64)v115);
  v104 = 256;
  sub_2AB1B50((__int64)v112);
  sub_2AB1B50((__int64)v110);
  sub_2ABCC20(v96, (__int64)v120, v72, v73, v74, v75);
  v97 = v121;
  sub_2ABCC20(v98, (__int64)v122, v121, v76, v77, v78);
  v99 = v123;
  sub_2C2B3B0(v110, v96);
  v111 = v97;
  sub_2C2B3B0(v112, v98);
  v113 = v99;
  sub_2C2B3B0(v115, v110);
  v116 = v111;
  sub_2C2B3B0(v117, v112);
  v118 = v113;
  sub_2C2B3B0(v105, v115);
  v106 = v116;
  sub_2C2B3B0(v107, v117);
  v108 = v118;
  sub_2AB1B50((__int64)v117);
  sub_2AB1B50((__int64)v115);
  v109 = 256;
  sub_2AB1B50((__int64)v112);
  sub_2AB1B50((__int64)v110);
  sub_2C2B3B0(v115, v100);
  v116 = v101;
  sub_2C2B3B0(v117, v102);
  v118 = v103;
  v119 = v104;
  sub_2C2B3B0(v110, v105);
  v111 = v106;
  sub_2C2B3B0(v112, v107);
  v113 = v108;
  v114 = v109;
  sub_2C2B3B0((_QWORD *)a1, v110);
  *(_WORD *)(a1 + 120) = v111;
  sub_2C2B3B0((_QWORD *)(a1 + 128), v112);
  *(_WORD *)(a1 + 248) = v113;
  *(_WORD *)(a1 + 264) = v114;
  sub_2C2B3B0((_QWORD *)(a1 + 272), v115);
  *(_WORD *)(a1 + 392) = v116;
  sub_2C2B3B0((_QWORD *)(a1 + 400), v117);
  *(_WORD *)(a1 + 520) = v118;
  *(_WORD *)(a1 + 536) = v119;
  sub_2AB1B50((__int64)v112);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v117);
  sub_2AB1B50((__int64)v115);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v105);
  sub_2AB1B50((__int64)v98);
  sub_2AB1B50((__int64)v96);
  sub_2AB1B50((__int64)v102);
  sub_2AB1B50((__int64)v100);
  sub_2AB1B50((__int64)v94);
  sub_2AB1B50((__int64)v92);
  sub_2AB1B50((__int64)v126);
  sub_2AB1B50((__int64)v124);
  sub_2AB1B50((__int64)v122);
  sub_2AB1B50((__int64)v120);
  sub_2AB1B50((__int64)v90);
  sub_2AB1B50((__int64)v88);
  return a1;
}
