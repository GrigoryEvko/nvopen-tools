// Function: sub_1602D10
// Address: 0x1602d10
//
__int64 __fastcall sub_1602D10(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  char *v3; // rbx
  const void *v4; // rsi
  size_t v5; // rdx
  char v7; // [rsp+0h] [rbp-280h] BYREF
  char *v8; // [rsp+8h] [rbp-278h]
  __int64 v9; // [rsp+10h] [rbp-270h]
  int v10; // [rsp+18h] [rbp-268h]
  char *v11; // [rsp+20h] [rbp-260h]
  __int64 v12; // [rsp+28h] [rbp-258h]
  int v13; // [rsp+30h] [rbp-250h]
  char *v14; // [rsp+38h] [rbp-248h]
  __int64 v15; // [rsp+40h] [rbp-240h]
  int v16; // [rsp+48h] [rbp-238h]
  const char *v17; // [rsp+50h] [rbp-230h]
  __int64 v18; // [rsp+58h] [rbp-228h]
  int v19; // [rsp+60h] [rbp-220h]
  char *v20; // [rsp+68h] [rbp-218h]
  __int64 v21; // [rsp+70h] [rbp-210h]
  int v22; // [rsp+78h] [rbp-208h]
  const char *v23; // [rsp+80h] [rbp-200h]
  __int64 v24; // [rsp+88h] [rbp-1F8h]
  int v25; // [rsp+90h] [rbp-1F0h]
  const char *v26; // [rsp+98h] [rbp-1E8h]
  __int64 v27; // [rsp+A0h] [rbp-1E0h]
  int v28; // [rsp+A8h] [rbp-1D8h]
  const char *v29; // [rsp+B0h] [rbp-1D0h]
  __int64 v30; // [rsp+B8h] [rbp-1C8h]
  int v31; // [rsp+C0h] [rbp-1C0h]
  char *v32; // [rsp+C8h] [rbp-1B8h]
  __int64 v33; // [rsp+D0h] [rbp-1B0h]
  int v34; // [rsp+D8h] [rbp-1A8h]
  const char *v35; // [rsp+E0h] [rbp-1A0h]
  __int64 v36; // [rsp+E8h] [rbp-198h]
  int v37; // [rsp+F0h] [rbp-190h]
  const char *v38; // [rsp+F8h] [rbp-188h]
  __int64 v39; // [rsp+100h] [rbp-180h]
  int v40; // [rsp+108h] [rbp-178h]
  const char *v41; // [rsp+110h] [rbp-170h]
  __int64 v42; // [rsp+118h] [rbp-168h]
  int v43; // [rsp+120h] [rbp-160h]
  char *v44; // [rsp+128h] [rbp-158h]
  __int64 v45; // [rsp+130h] [rbp-150h]
  int v46; // [rsp+138h] [rbp-148h]
  const char *v47; // [rsp+140h] [rbp-140h]
  __int64 v48; // [rsp+148h] [rbp-138h]
  int v49; // [rsp+150h] [rbp-130h]
  const char *v50; // [rsp+158h] [rbp-128h]
  __int64 v51; // [rsp+160h] [rbp-120h]
  int v52; // [rsp+168h] [rbp-118h]
  char *v53; // [rsp+170h] [rbp-110h]
  __int64 v54; // [rsp+178h] [rbp-108h]
  int v55; // [rsp+180h] [rbp-100h]
  const char *v56; // [rsp+188h] [rbp-F8h]
  __int64 v57; // [rsp+190h] [rbp-F0h]
  int v58; // [rsp+198h] [rbp-E8h]
  char *v59; // [rsp+1A0h] [rbp-E0h]
  __int64 v60; // [rsp+1A8h] [rbp-D8h]
  int v61; // [rsp+1B0h] [rbp-D0h]
  const char *v62; // [rsp+1B8h] [rbp-C8h]
  __int64 v63; // [rsp+1C0h] [rbp-C0h]
  int v64; // [rsp+1C8h] [rbp-B8h]
  char *v65; // [rsp+1D0h] [rbp-B0h]
  __int64 v66; // [rsp+1D8h] [rbp-A8h]
  int v67; // [rsp+1E0h] [rbp-A0h]
  char *v68; // [rsp+1E8h] [rbp-98h]
  __int64 v69; // [rsp+1F0h] [rbp-90h]
  int v70; // [rsp+1F8h] [rbp-88h]
  const char *v71; // [rsp+200h] [rbp-80h]
  __int64 v72; // [rsp+208h] [rbp-78h]
  int v73; // [rsp+210h] [rbp-70h]
  char *v74; // [rsp+218h] [rbp-68h]
  __int64 v75; // [rsp+220h] [rbp-60h]
  int v76; // [rsp+228h] [rbp-58h]
  const char *v77; // [rsp+230h] [rbp-50h]
  __int64 v78; // [rsp+238h] [rbp-48h]
  int v79; // [rsp+240h] [rbp-40h]
  const char *v80; // [rsp+248h] [rbp-38h]
  __int64 v81; // [rsp+250h] [rbp-30h]
  char v82; // [rsp+258h] [rbp-28h] BYREF

  v1 = sub_22077B0(2976);
  v2 = v1;
  if ( v1 )
    sub_1604350(v1, a1);
  *a1 = v2;
  v8 = "dbg";
  v3 = &v7;
  v11 = "tbaa";
  v14 = "prof";
  v17 = "fpmath";
  v20 = "range";
  v23 = "tbaa.struct";
  v26 = "invariant.load";
  v29 = "alias.scope";
  v32 = "noalias";
  v35 = "nontemporal";
  v38 = "llvm.mem.parallel_loop_access";
  v9 = 3;
  v10 = 1;
  v12 = 4;
  v13 = 2;
  v15 = 4;
  v16 = 3;
  v18 = 6;
  v19 = 4;
  v21 = 5;
  v22 = 5;
  v24 = 11;
  v25 = 6;
  v27 = 14;
  v28 = 7;
  v30 = 11;
  v31 = 8;
  v33 = 7;
  v34 = 9;
  v36 = 11;
  v37 = 10;
  v39 = 29;
  v41 = "nonnull";
  v44 = "dereferenceable";
  v47 = "dereferenceable_or_null";
  v50 = "make.implicit";
  v53 = "unpredictable";
  v56 = "invariant.group";
  v59 = "align";
  v62 = "llvm.loop";
  v65 = "type";
  v68 = "section_prefix";
  v71 = "absolute_symbol";
  v40 = 11;
  v42 = 7;
  v43 = 12;
  v45 = 15;
  v46 = 13;
  v48 = 23;
  v49 = 14;
  v51 = 13;
  v52 = 15;
  v54 = 13;
  v55 = 16;
  v57 = 15;
  v58 = 17;
  v60 = 5;
  v61 = 18;
  v63 = 9;
  v64 = 19;
  v66 = 4;
  v67 = 20;
  v69 = 14;
  v70 = 21;
  v72 = 15;
  v74 = "associated";
  v77 = "callees";
  v73 = 22;
  v75 = 10;
  v76 = 23;
  v78 = 7;
  v79 = 24;
  v80 = "irr_loop";
  v81 = 8;
  do
  {
    v4 = (const void *)*((_QWORD *)v3 + 1);
    v5 = *((_QWORD *)v3 + 2);
    v3 += 24;
    sub_1602B80(a1, v4, v5);
  }
  while ( &v82 != v3 );
  sub_16052C0(*a1, "deopt", 5);
  sub_16052C0(*a1, "funclet", 7);
  sub_16052C0(*a1, "gc-transition", 13);
  sub_1605610(*a1, "singlethread", 12);
  sub_1605610(*a1, byte_3F871B3, 0);
  sub_1605610(*a1, "CTA", 3);
  sub_1605610(*a1, "GPU", 3);
  return sub_1605610(*a1, "cluster", 7);
}
