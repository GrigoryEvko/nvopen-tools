// Function: sub_3469040
// Address: 0x3469040
//
__int64 __fastcall sub_3469040(
        __m128i a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __int64 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11)
{
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // ebx
  unsigned __int64 v18; // rax
  __int64 v19; // r9
  unsigned __int8 *v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r15
  __int128 v23; // rax
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  __int128 v27; // rax
  __int64 v28; // r9
  __int128 v29; // rax
  __int128 v30; // rax
  __int64 v31; // r9
  __int128 v32; // rax
  unsigned int v33; // ebx
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  __int128 v37; // rax
  __int64 v38; // r9
  __int128 v39; // rax
  __int64 v40; // r9
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r15
  unsigned __int8 *v46; // r14
  __int64 v47; // r9
  __int128 v48; // rax
  __int64 v49; // r9
  __int128 v50; // rax
  __int64 v51; // r9
  __int128 v52; // rax
  __int64 v53; // r15
  __int64 v54; // r14
  __int64 v55; // r9
  __int128 v56; // rax
  __int64 v57; // r9
  __int128 v58; // rax
  __int64 v59; // r9
  unsigned __int8 *v60; // rax
  __int64 v61; // r8
  int v62; // edx
  __int64 v63; // r9
  unsigned __int8 *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r15
  unsigned __int8 *v67; // r14
  __int64 v68; // r9
  __int128 v69; // rax
  __int64 v70; // r9
  __int64 v71; // r9
  int v72; // edx
  __int64 result; // rax
  unsigned __int8 *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r15
  unsigned __int8 *v77; // r14
  __int64 v78; // r9
  __int128 v79; // rax
  __int64 v80; // r9
  __int128 v81; // rax
  __int64 v82; // r9
  unsigned int v83; // edx
  __int128 v84; // [rsp-20h] [rbp-130h]
  __int128 v85; // [rsp-10h] [rbp-120h]
  __int128 v86; // [rsp-10h] [rbp-120h]
  __int128 v87; // [rsp-10h] [rbp-120h]
  __int128 v88; // [rsp-10h] [rbp-120h]
  __int128 v89; // [rsp-10h] [rbp-120h]
  __int128 v90; // [rsp-10h] [rbp-120h]
  __int128 v91; // [rsp-10h] [rbp-120h]
  __int128 v92; // [rsp+0h] [rbp-110h]
  __int128 v95; // [rsp+20h] [rbp-F0h]
  __int128 v96; // [rsp+30h] [rbp-E0h]
  __int128 v97; // [rsp+30h] [rbp-E0h]
  __int128 v99; // [rsp+40h] [rbp-D0h]
  __int128 v100; // [rsp+50h] [rbp-C0h]
  __int128 v101; // [rsp+50h] [rbp-C0h]
  __int128 v102; // [rsp+50h] [rbp-C0h]
  __int128 v103; // [rsp+60h] [rbp-B0h]
  __int128 v104; // [rsp+60h] [rbp-B0h]
  __int128 v105; // [rsp+70h] [rbp-A0h]
  unsigned int v106; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v107; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v108; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v109; // [rsp+C8h] [rbp-48h]
  __int64 v110; // [rsp+D0h] [rbp-40h]
  __int64 v111; // [rsp+D8h] [rbp-38h]

  v12 = *(_QWORD *)(a8 + 48) + 16LL * DWORD2(a8);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v106) = v13;
  v107 = v14;
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      BUG();
    v16 = 16LL * (v13 - 1);
    v15 = *(_QWORD *)&byte_444C4A0[v16];
    LOBYTE(v16) = byte_444C4A0[v16 + 8];
  }
  else
  {
    v15 = sub_3007260((__int64)&v106);
    v110 = v15;
    v111 = v16;
  }
  v108 = v15;
  LOBYTE(v109) = v16;
  v109 = sub_CA1930(&v108);
  v17 = v109 >> 1;
  if ( v109 > 0x40 )
    sub_C43690((__int64)&v108, 0, 0);
  else
    v108 = 0;
  if ( v17 )
  {
    if ( v17 > 0x40 )
    {
      sub_C43C90(&v108, 0, v17);
    }
    else
    {
      v18 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17);
      if ( v109 > 0x40 )
        *(_QWORD *)v108 |= v18;
      else
        v108 |= v18;
    }
  }
  v20 = sub_34007B0((__int64)a3, (__int64)&v108, a4, v106, v107, 0, a1, 0);
  v22 = v21;
  if ( v109 > 0x40 && v108 )
    j_j___libc_free_0_0(v108);
  *((_QWORD *)&v85 + 1) = v22;
  *(_QWORD *)&v85 = v20;
  *(_QWORD *)&v23 = sub_3406EB0(a3, 0xBAu, a4, v106, v107, v19, a8, v85);
  *((_QWORD *)&v86 + 1) = v22;
  *(_QWORD *)&v86 = v20;
  v103 = v23;
  *(_QWORD *)&v25 = sub_3406EB0(a3, 0xBAu, a4, v106, v107, v24, a9, v86);
  v100 = v25;
  *(_QWORD *)&v27 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v26, v103, v25);
  *((_QWORD *)&v87 + 1) = v22;
  *(_QWORD *)&v87 = v20;
  v96 = v27;
  *(_QWORD *)&v29 = sub_3406EB0(a3, 0xBAu, a4, v106, v107, v28, v27, v87);
  v95 = v29;
  *(_QWORD *)&v30 = sub_3400E40((__int64)a3, v17, v106, v107, a4, a1);
  v105 = v30;
  *(_QWORD *)&v32 = sub_3406EB0(a3, 0xC0u, a4, v106, v107, v31, v96, v30);
  v33 = (a5 == 0) + 191;
  v92 = v32;
  *(_QWORD *)&v35 = sub_3406EB0(a3, v33, a4, v106, v107, v34, a8, v105);
  v99 = v35;
  *(_QWORD *)&v37 = sub_3406EB0(a3, v33, a4, v106, v107, v36, a9, v105);
  v97 = v37;
  *(_QWORD *)&v39 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v38, v99, v100);
  *(_QWORD *)&v41 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v40, v39, v92);
  *((_QWORD *)&v88 + 1) = v22;
  *(_QWORD *)&v88 = v20;
  v101 = v41;
  v43 = sub_3406EB0(a3, 0xBAu, a4, v106, v107, v42, v41, v88);
  v45 = v44;
  v46 = v43;
  *(_QWORD *)&v48 = sub_3406EB0(a3, v33, a4, v106, v107, v47, v101, v105);
  v102 = v48;
  *(_QWORD *)&v50 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v49, v103, v97);
  *((_QWORD *)&v89 + 1) = v45;
  *(_QWORD *)&v89 = v46;
  *(_QWORD *)&v52 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v51, v50, v89);
  v53 = *((_QWORD *)&v52 + 1);
  v54 = v52;
  *(_QWORD *)&v56 = sub_3406EB0(a3, v33, a4, v106, v107, v55, v52, v105);
  *((_QWORD *)&v84 + 1) = v53;
  *(_QWORD *)&v84 = v54;
  v104 = v56;
  *(_QWORD *)&v58 = sub_3406EB0(a3, 0xBEu, a4, v106, v107, v57, v84, v105);
  v60 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v59, v95, v58);
  v61 = v107;
  *(_QWORD *)a6 = v60;
  *(_DWORD *)(a6 + 8) = v62;
  v64 = sub_3406EB0(a3, 0x38u, a4, v106, v61, v63, v102, v104);
  v66 = v65;
  v67 = v64;
  *(_QWORD *)&v69 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v68, v99, v97);
  *((_QWORD *)&v90 + 1) = v66;
  *(_QWORD *)&v90 = v67;
  *(_QWORD *)a7 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v70, v69, v90);
  *(_DWORD *)(a7 + 8) = v72;
  result = a10;
  if ( (_QWORD)a10 )
  {
    v74 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v71, a9, a10);
    v76 = v75;
    v77 = v74;
    *(_QWORD *)&v79 = sub_3406EB0(a3, 0x3Au, a4, v106, v107, v78, a11, a8);
    *((_QWORD *)&v91 + 1) = v76;
    *(_QWORD *)&v91 = v77;
    *(_QWORD *)&v81 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v80, v79, v91);
    *(_QWORD *)a7 = sub_3406EB0(a3, 0x38u, a4, v106, v107, v82, *(_OWORD *)a7, v81);
    result = v83;
    *(_DWORD *)(a7 + 8) = v83;
  }
  return result;
}
