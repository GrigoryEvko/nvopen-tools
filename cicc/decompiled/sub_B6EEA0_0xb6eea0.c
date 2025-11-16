// Function: sub_B6EEA0
// Address: 0xb6eea0
//
__int64 __fastcall sub_B6EEA0(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  int *v4; // rbx
  const void *v5; // rsi
  size_t v6; // rdx
  int v7; // r15d
  __int64 v8; // rdi
  __int64 v9; // rdx
  char *v10; // rsi
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-428h] BYREF
  int v14; // [rsp+10h] [rbp-420h] BYREF
  char *v15; // [rsp+18h] [rbp-418h]
  __int64 v16; // [rsp+20h] [rbp-410h]
  int v17; // [rsp+28h] [rbp-408h]
  char *v18; // [rsp+30h] [rbp-400h]
  __int64 v19; // [rsp+38h] [rbp-3F8h]
  int v20; // [rsp+40h] [rbp-3F0h]
  char *v21; // [rsp+48h] [rbp-3E8h]
  __int64 v22; // [rsp+50h] [rbp-3E0h]
  int v23; // [rsp+58h] [rbp-3D8h]
  const char *v24; // [rsp+60h] [rbp-3D0h]
  __int64 v25; // [rsp+68h] [rbp-3C8h]
  int v26; // [rsp+70h] [rbp-3C0h]
  char *v27; // [rsp+78h] [rbp-3B8h]
  __int64 v28; // [rsp+80h] [rbp-3B0h]
  int v29; // [rsp+88h] [rbp-3A8h]
  const char *v30; // [rsp+90h] [rbp-3A0h]
  __int64 v31; // [rsp+98h] [rbp-398h]
  int v32; // [rsp+A0h] [rbp-390h]
  const char *v33; // [rsp+A8h] [rbp-388h]
  __int64 v34; // [rsp+B0h] [rbp-380h]
  int v35; // [rsp+B8h] [rbp-378h]
  const char *v36; // [rsp+C0h] [rbp-370h]
  __int64 v37; // [rsp+C8h] [rbp-368h]
  int v38; // [rsp+D0h] [rbp-360h]
  char *v39; // [rsp+D8h] [rbp-358h]
  __int64 v40; // [rsp+E0h] [rbp-350h]
  int v41; // [rsp+E8h] [rbp-348h]
  const char *v42; // [rsp+F0h] [rbp-340h]
  __int64 v43; // [rsp+F8h] [rbp-338h]
  int v44; // [rsp+100h] [rbp-330h]
  const char *v45; // [rsp+108h] [rbp-328h]
  __int64 v46; // [rsp+110h] [rbp-320h]
  int v47; // [rsp+118h] [rbp-318h]
  const char *v48; // [rsp+120h] [rbp-310h]
  __int64 v49; // [rsp+128h] [rbp-308h]
  int v50; // [rsp+130h] [rbp-300h]
  char *v51; // [rsp+138h] [rbp-2F8h]
  __int64 v52; // [rsp+140h] [rbp-2F0h]
  int v53; // [rsp+148h] [rbp-2E8h]
  const char *v54; // [rsp+150h] [rbp-2E0h]
  __int64 v55; // [rsp+158h] [rbp-2D8h]
  int v56; // [rsp+160h] [rbp-2D0h]
  const char *v57; // [rsp+168h] [rbp-2C8h]
  __int64 v58; // [rsp+170h] [rbp-2C0h]
  int v59; // [rsp+178h] [rbp-2B8h]
  char *v60; // [rsp+180h] [rbp-2B0h]
  __int64 v61; // [rsp+188h] [rbp-2A8h]
  int v62; // [rsp+190h] [rbp-2A0h]
  const char *v63; // [rsp+198h] [rbp-298h]
  __int64 v64; // [rsp+1A0h] [rbp-290h]
  int v65; // [rsp+1A8h] [rbp-288h]
  char *v66; // [rsp+1B0h] [rbp-280h]
  __int64 v67; // [rsp+1B8h] [rbp-278h]
  int v68; // [rsp+1C0h] [rbp-270h]
  const char *v69; // [rsp+1C8h] [rbp-268h]
  __int64 v70; // [rsp+1D0h] [rbp-260h]
  int v71; // [rsp+1D8h] [rbp-258h]
  char *v72; // [rsp+1E0h] [rbp-250h]
  __int64 v73; // [rsp+1E8h] [rbp-248h]
  int v74; // [rsp+1F0h] [rbp-240h]
  char *v75; // [rsp+1F8h] [rbp-238h]
  __int64 v76; // [rsp+200h] [rbp-230h]
  int v77; // [rsp+208h] [rbp-228h]
  const char *v78; // [rsp+210h] [rbp-220h]
  __int64 v79; // [rsp+218h] [rbp-218h]
  int v80; // [rsp+220h] [rbp-210h]
  char *v81; // [rsp+228h] [rbp-208h]
  __int64 v82; // [rsp+230h] [rbp-200h]
  int v83; // [rsp+238h] [rbp-1F8h]
  const char *v84; // [rsp+240h] [rbp-1F0h]
  __int64 v85; // [rsp+248h] [rbp-1E8h]
  int v86; // [rsp+250h] [rbp-1E0h]
  const char *v87; // [rsp+258h] [rbp-1D8h]
  __int64 v88; // [rsp+260h] [rbp-1D0h]
  int v89; // [rsp+268h] [rbp-1C8h]
  const char *v90; // [rsp+270h] [rbp-1C0h]
  __int64 v91; // [rsp+278h] [rbp-1B8h]
  int v92; // [rsp+280h] [rbp-1B0h]
  char *v93; // [rsp+288h] [rbp-1A8h]
  __int64 v94; // [rsp+290h] [rbp-1A0h]
  int v95; // [rsp+298h] [rbp-198h]
  const char *v96; // [rsp+2A0h] [rbp-190h]
  __int64 v97; // [rsp+2A8h] [rbp-188h]
  int v98; // [rsp+2B0h] [rbp-180h]
  const char *v99; // [rsp+2B8h] [rbp-178h]
  __int64 v100; // [rsp+2C0h] [rbp-170h]
  int v101; // [rsp+2C8h] [rbp-168h]
  const char *v102; // [rsp+2D0h] [rbp-160h]
  __int64 v103; // [rsp+2D8h] [rbp-158h]
  int v104; // [rsp+2E0h] [rbp-150h]
  char *v105; // [rsp+2E8h] [rbp-148h]
  __int64 v106; // [rsp+2F0h] [rbp-140h]
  int v107; // [rsp+2F8h] [rbp-138h]
  const char *v108; // [rsp+300h] [rbp-130h]
  __int64 v109; // [rsp+308h] [rbp-128h]
  int v110; // [rsp+310h] [rbp-120h]
  const char *v111; // [rsp+318h] [rbp-118h]
  __int64 v112; // [rsp+320h] [rbp-110h]
  int v113; // [rsp+328h] [rbp-108h]
  char *v114; // [rsp+330h] [rbp-100h]
  __int64 v115; // [rsp+338h] [rbp-F8h]
  int v116; // [rsp+340h] [rbp-F0h]
  char *v117; // [rsp+348h] [rbp-E8h]
  __int64 v118; // [rsp+350h] [rbp-E0h]
  int v119; // [rsp+358h] [rbp-D8h]
  char *v120; // [rsp+360h] [rbp-D0h]
  __int64 v121; // [rsp+368h] [rbp-C8h]
  int v122; // [rsp+370h] [rbp-C0h]
  const char *v123; // [rsp+378h] [rbp-B8h]
  __int64 v124; // [rsp+380h] [rbp-B0h]
  int v125; // [rsp+388h] [rbp-A8h]
  const char *v126; // [rsp+390h] [rbp-A0h]
  __int64 v127; // [rsp+398h] [rbp-98h]
  int v128; // [rsp+3A0h] [rbp-90h]
  const char *v129; // [rsp+3A8h] [rbp-88h]
  __int64 v130; // [rsp+3B0h] [rbp-80h]
  int v131; // [rsp+3B8h] [rbp-78h]
  const char *v132; // [rsp+3C0h] [rbp-70h]
  __int64 v133; // [rsp+3C8h] [rbp-68h]
  int v134; // [rsp+3D0h] [rbp-60h]
  const char *v135; // [rsp+3D8h] [rbp-58h]
  __int64 v136; // [rsp+3E0h] [rbp-50h]
  int v137; // [rsp+3E8h] [rbp-48h]
  const char *v138; // [rsp+3F0h] [rbp-40h]
  __int64 v139; // [rsp+3F8h] [rbp-38h]
  char v140; // [rsp+400h] [rbp-30h] BYREF

  v2 = sub_22077B0(3656);
  v3 = v2;
  if ( v2 )
    sub_B70800(v2, a1);
  *a1 = v3;
  v15 = "dbg";
  v4 = &v14;
  v18 = "tbaa";
  v21 = "prof";
  v24 = "fpmath";
  v27 = "range";
  v30 = "tbaa.struct";
  v33 = "invariant.load";
  v36 = "alias.scope";
  v39 = "noalias";
  v42 = "nontemporal";
  v14 = 0;
  v16 = 3;
  v17 = 1;
  v19 = 4;
  v20 = 2;
  v22 = 4;
  v23 = 3;
  v25 = 6;
  v26 = 4;
  v28 = 5;
  v29 = 5;
  v31 = 11;
  v32 = 6;
  v34 = 14;
  v35 = 7;
  v37 = 11;
  v38 = 8;
  v40 = 7;
  v41 = 9;
  v43 = 11;
  v44 = 10;
  v45 = "llvm.mem.parallel_loop_access";
  v48 = "nonnull";
  v51 = "dereferenceable";
  v54 = "dereferenceable_or_null";
  v57 = "make.implicit";
  v60 = "unpredictable";
  v63 = "invariant.group";
  v66 = "align";
  v69 = "llvm.loop";
  v72 = "type";
  v75 = "section_prefix";
  v46 = 29;
  v47 = 11;
  v49 = 7;
  v50 = 12;
  v52 = 15;
  v53 = 13;
  v55 = 23;
  v56 = 14;
  v58 = 13;
  v59 = 15;
  v61 = 13;
  v62 = 16;
  v64 = 15;
  v65 = 17;
  v67 = 5;
  v68 = 18;
  v70 = 9;
  v71 = 19;
  v73 = 4;
  v74 = 20;
  v76 = 14;
  v77 = 21;
  v78 = "absolute_symbol";
  v81 = "associated";
  v84 = "callees";
  v87 = "irr_loop";
  v90 = "llvm.access.group";
  v93 = "callback";
  v96 = "llvm.preserve.access.index";
  v99 = "vcall_visibility";
  v102 = "noundef";
  v105 = "annotation";
  v108 = "nosanitize";
  v79 = 15;
  v80 = 22;
  v82 = 10;
  v83 = 23;
  v85 = 7;
  v86 = 24;
  v88 = 8;
  v89 = 25;
  v91 = 17;
  v92 = 26;
  v94 = 8;
  v95 = 27;
  v97 = 26;
  v98 = 28;
  v100 = 16;
  v101 = 29;
  v103 = 7;
  v104 = 30;
  v106 = 10;
  v107 = 31;
  v109 = 10;
  v110 = 32;
  v111 = "func_sanitize";
  v114 = "exclude";
  v117 = "memprof";
  v120 = "callsite";
  v123 = "kcfi_type";
  v126 = "pcsections";
  v129 = "DIAssignID";
  v132 = "coro.outside.frame";
  v135 = "mmra";
  v112 = 13;
  v113 = 33;
  v115 = 7;
  v116 = 34;
  v118 = 7;
  v119 = 35;
  v121 = 8;
  v122 = 36;
  v124 = 9;
  v125 = 37;
  v127 = 10;
  v128 = 38;
  v130 = 10;
  v131 = 39;
  v133 = 18;
  v134 = 40;
  v136 = 4;
  v137 = 41;
  v138 = "noalias.addrspace";
  v139 = 17;
  do
  {
    v5 = (const void *)*((_QWORD *)v4 + 1);
    v6 = *((_QWORD *)v4 + 2);
    v4 += 6;
    sub_B6ED60(a1, v5, v6);
  }
  while ( v4 != (int *)&v140 );
  v7 = 1;
  while ( 2 )
  {
    v8 = *a1;
    switch ( v7 )
    {
      case 2:
        v9 = 7;
        v10 = "funclet";
        goto LABEL_8;
      case 3:
        v9 = 13;
        v10 = "gc-transition";
        goto LABEL_8;
      case 4:
        v9 = 13;
        v10 = "cfguardtarget";
        goto LABEL_8;
      case 5:
        v9 = 12;
        v10 = "preallocated";
        goto LABEL_8;
      case 6:
        v9 = 7;
        v10 = "gc-live";
        goto LABEL_8;
      case 7:
        v9 = 22;
        v10 = "clang.arc.attachedcall";
        goto LABEL_8;
      case 8:
        v9 = 7;
        v10 = "ptrauth";
        goto LABEL_8;
      case 9:
        v9 = 4;
        v10 = "kcfi";
LABEL_8:
        sub_B71A20(v8, v10, v9);
        goto LABEL_9;
      case 10:
        sub_B71A20(v8, "convergencectrl", 15);
        goto LABEL_13;
      default:
        sub_B71A20(v8, "deopt", 5);
        if ( v7 != 10 )
        {
LABEL_9:
          ++v7;
          continue;
        }
LABEL_13:
        sub_B71D20(*a1, "singlethread", 12);
        sub_B71D20(*a1, byte_3F871B3, 0);
        sub_B71D20(*a1, "CTA", 3);
        sub_B71D20(*a1, "GPU", 3);
        sub_B71D20(*a1, "cluster", 7);
        v11 = sub_22077B0(32);
        if ( v11 )
        {
          *(_QWORD *)(v11 + 8) = 0;
          *(_BYTE *)(v11 + 16) = 0;
          *(_QWORD *)(v11 + 24) = 0;
          *(_QWORD *)v11 = &unk_49DA3B0;
        }
        v13 = v11;
        result = sub_B6E8B0(a1, &v13, 0);
        if ( v13 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
        return result;
    }
  }
}
