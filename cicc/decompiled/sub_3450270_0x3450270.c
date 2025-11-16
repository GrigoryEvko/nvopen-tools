// Function: sub_3450270
// Address: 0x3450270
//
bool __fastcall sub_3450270(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // rax
  unsigned __int16 v11; // cx
  __int16 *v12; // rax
  unsigned __int16 v13; // r8
  __int64 v14; // rbx
  unsigned int v15; // r14d
  bool result; // al
  int v17; // edx
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int128 v24; // rax
  __int128 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  unsigned __int8 *v28; // rax
  unsigned int v29; // edx
  __int128 v30; // rax
  int v31; // r9d
  __int128 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  __int128 v37; // rax
  __int64 v38; // r9
  __int128 v39; // rax
  __int128 v40; // rax
  __int64 v41; // r9
  __int128 v42; // rax
  __int64 v43; // r9
  unsigned __int8 *v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // edx
  __int128 v47; // rax
  __int64 v48; // r9
  __int128 v49; // rax
  __int64 v50; // r9
  unsigned __int8 *v51; // rax
  __int64 v52; // rdx
  unsigned int v53; // edx
  __int64 v54; // r9
  unsigned __int8 *v55; // rax
  unsigned int v56; // edx
  __int128 v57; // rax
  __int64 v58; // r9
  __int128 v59; // rax
  __int64 v60; // r9
  unsigned __int8 *v61; // rax
  unsigned int v62; // edx
  __int128 v63; // rax
  __int64 v64; // r9
  __int64 v65; // rdx
  __int128 v66; // rax
  unsigned __int16 *v67; // rsi
  __int64 v68; // r9
  unsigned int v69; // edx
  __int64 v70; // r9
  __int128 v71; // rax
  __int64 v72; // r9
  __int128 v73; // rax
  __int64 v74; // rdx
  unsigned __int8 *v75; // r14
  __int64 v76; // rdx
  __int64 v77; // r15
  __int128 v78; // rax
  __int64 v79; // r9
  int v80; // edx
  __int128 v81; // [rsp-40h] [rbp-1A0h]
  __int128 v82; // [rsp-30h] [rbp-190h]
  __int128 v83; // [rsp-20h] [rbp-180h]
  __int64 v84; // [rsp+10h] [rbp-150h]
  unsigned int v85; // [rsp+18h] [rbp-148h]
  __int128 v86; // [rsp+20h] [rbp-140h]
  __int128 v87; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v88; // [rsp+30h] [rbp-130h]
  __int128 v89; // [rsp+40h] [rbp-120h]
  __int128 v90; // [rsp+40h] [rbp-120h]
  __int128 v91; // [rsp+40h] [rbp-120h]
  __int64 v92; // [rsp+50h] [rbp-110h]
  __int64 v93; // [rsp+58h] [rbp-108h]
  __int128 v94; // [rsp+60h] [rbp-100h]
  __int128 v95; // [rsp+60h] [rbp-100h]
  __int128 v96; // [rsp+70h] [rbp-F0h]
  __int128 v97; // [rsp+70h] [rbp-F0h]
  __int128 v98; // [rsp+80h] [rbp-E0h]
  __int128 v99; // [rsp+80h] [rbp-E0h]
  unsigned int v100; // [rsp+90h] [rbp-D0h]
  unsigned int v101; // [rsp+90h] [rbp-D0h]
  __int128 v103; // [rsp+A0h] [rbp-C0h]
  unsigned __int16 v104; // [rsp+B0h] [rbp-B0h]
  bool v105; // [rsp+B0h] [rbp-B0h]
  unsigned int v106; // [rsp+B0h] [rbp-B0h]
  __int64 v107; // [rsp+B8h] [rbp-A8h]
  unsigned __int8 *v108; // [rsp+D0h] [rbp-90h]
  unsigned __int16 v109; // [rsp+100h] [rbp-60h] BYREF
  __int64 v110; // [rsp+108h] [rbp-58h]
  __int64 v111; // [rsp+110h] [rbp-50h] BYREF
  int v112; // [rsp+118h] [rbp-48h]
  unsigned __int64 v113; // [rsp+120h] [rbp-40h] BYREF
  unsigned int v114; // [rsp+128h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 > 239 )
  {
    v7 = (unsigned int)(v6 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v7 = 40;
    if ( v6 <= 237 )
      v7 = (unsigned int)(v6 - 101) < 0x30 ? 0x28 : 0;
  }
  v8 = *(_QWORD *)(a2 + 80);
  v9 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v7));
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v7) + 48LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v7 + 8);
  v11 = *(_WORD *)v10;
  v110 = *(_QWORD *)(v10 + 8);
  v12 = *(__int16 **)(a2 + 48);
  v109 = v11;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v111 = v8;
  v15 = v13;
  if ( v8 )
  {
    v104 = v13;
    sub_B96E90((__int64)&v111, v8, 1);
    v11 = v109;
    v13 = v104;
  }
  v112 = *(_DWORD *)(a2 + 72);
  result = v13 != 8 || v11 != 12;
  if ( result )
  {
    result = 0;
    goto LABEL_8;
  }
  v17 = *(_DWORD *)(a2 + 24);
  if ( v17 > 239 )
  {
    if ( (unsigned int)(v17 - 242) <= 1 )
      goto LABEL_8;
  }
  else if ( v17 > 237 || (unsigned int)(v17 - 101) <= 0x2F )
  {
    goto LABEL_8;
  }
  v18 = sub_327FF20(&v109, v8);
  v20 = v19;
  v106 = v18;
  v21 = sub_2E79000(*(__int64 **)(a4 + 40));
  v22 = sub_2FE6750(a1, v106, v20, v21);
  v92 = v23;
  v100 = v22;
  *(_QWORD *)&v24 = sub_3400BD0(a4, 2139095040, (__int64)&v111, v106, v20, 0, v9, 0);
  v94 = v24;
  *(_QWORD *)&v25 = sub_3400BD0(a4, 23, (__int64)&v111, v106, v20, 0, v9, 0);
  v98 = v25;
  *(_QWORD *)&v89 = sub_3400BD0(a4, 127, (__int64)&v111, v106, v20, 0, v9, 0);
  *((_QWORD *)&v89 + 1) = v26;
  v114 = 32;
  v113 = 0x80000000LL;
  *(_QWORD *)&v87 = sub_34007B0(a4, (__int64)&v113, (__int64)&v111, v106, v20, 0, v9, 0);
  *((_QWORD *)&v87 + 1) = v27;
  if ( v114 > 0x40 && v113 )
    j_j___libc_free_0_0(v113);
  v28 = sub_3400BD0(a4, 31, (__int64)&v111, v106, v20, 0, v9, 0);
  v85 = v29;
  v84 = (__int64)v28;
  *(_QWORD *)&v30 = sub_3400BD0(a4, 0x7FFFFF, (__int64)&v111, v106, v20, 0, v9, 0);
  v86 = v30;
  *(_QWORD *)&v32 = sub_33FAF80(a4, 234, (__int64)&v111, v106, v20, v31, v9);
  v103 = v32;
  *(_QWORD *)&v33 = sub_33FB310(a4, v98, DWORD2(v98), (__int64)&v111, v100, v92, v9);
  v96 = v33;
  *(_QWORD *)&v35 = sub_3406EB0((_QWORD *)a4, 0xBAu, (__int64)&v111, v106, v20, v34, v103, v94);
  *(_QWORD *)&v37 = sub_3406EB0((_QWORD *)a4, 0xC0u, (__int64)&v111, v106, v20, v36, v35, v96);
  *(_QWORD *)&v39 = sub_3406EB0((_QWORD *)a4, 0x39u, (__int64)&v111, v106, v20, v38, v37, v89);
  v95 = v39;
  *(_QWORD *)&v40 = sub_33FB310(a4, v84, v85, (__int64)&v111, v100, v92, v9);
  v97 = v40;
  *(_QWORD *)&v42 = sub_3406EB0((_QWORD *)a4, 0xBAu, (__int64)&v111, v106, v20, v41, v103, v87);
  v44 = sub_3406EB0((_QWORD *)a4, 0xBFu, (__int64)&v111, v106, v20, v43, v42, v97);
  *((_QWORD *)&v97 + 1) = v45;
  *(_QWORD *)&v97 = sub_33FB160(a4, (__int64)v44, v45, (__int64)&v111, v15, v14, v9);
  *((_QWORD *)&v97 + 1) = v46 | *((_QWORD *)&v97 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v47 = sub_3400BD0(a4, 0x800000, (__int64)&v111, v106, v20, 0, v9, 0);
  v90 = v47;
  *(_QWORD *)&v49 = sub_3406EB0((_QWORD *)a4, 0xBAu, (__int64)&v111, v106, v20, v48, v103, v86);
  v51 = sub_3406EB0((_QWORD *)a4, 0xBBu, (__int64)&v111, v106, v20, v50, v49, v90);
  *((_QWORD *)&v103 + 1) = v52;
  *(_QWORD *)&v103 = sub_33FB310(a4, (__int64)v51, v52, (__int64)&v111, v15, v14, v9);
  *((_QWORD *)&v103 + 1) = v53 | *((_QWORD *)&v103 + 1) & 0xFFFFFFFF00000000LL;
  v55 = sub_3406EB0((_QWORD *)a4, 0x39u, (__int64)&v111, v106, v20, v54, v98, v95);
  *(_QWORD *)&v57 = sub_33FB310(a4, (__int64)v55, v56, (__int64)&v111, v100, v92, v9);
  *(_QWORD *)&v59 = sub_3406EB0((_QWORD *)a4, 0xC0u, (__int64)&v111, v15, v14, v58, v103, v57);
  v91 = v59;
  v61 = sub_3406EB0((_QWORD *)a4, 0x39u, (__int64)&v111, v106, v20, v60, v95, v98);
  *(_QWORD *)&v63 = sub_33FB310(a4, (__int64)v61, v62, (__int64)&v111, v100, v92, v9);
  v88 = sub_3406EB0((_QWORD *)a4, 0xBEu, (__int64)&v111, v15, v14, v64, v103, v63);
  v101 = v65;
  v93 = v65;
  *(_QWORD *)&v66 = sub_33ED040((_QWORD *)a4, 0x12u);
  v67 = (unsigned __int16 *)(*((_QWORD *)v88 + 6) + 16LL * v101);
  *((_QWORD *)&v83 + 1) = v93;
  *(_QWORD *)&v83 = v88;
  v108 = sub_33FC1D0((_QWORD *)a4, 207, (__int64)&v111, *v67, *((_QWORD *)v67 + 1), v68, v95, v98, v83, v91, v66);
  *(_QWORD *)&v71 = sub_3406EB0(
                      (_QWORD *)a4,
                      0xBCu,
                      (__int64)&v111,
                      v15,
                      v14,
                      v70,
                      __PAIR128__(v69 | *((_QWORD *)&v103 + 1) & 0xFFFFFFFF00000000LL, (unsigned __int64)v108),
                      v97);
  *(_QWORD *)&v73 = sub_3406EB0((_QWORD *)a4, 0x39u, (__int64)&v111, v15, v14, v72, v71, v97);
  v99 = v73;
  *(_QWORD *)&v103 = sub_3400BD0(a4, 0, (__int64)&v111, v15, v14, 0, v9, 0);
  v107 = v74;
  v75 = sub_3400BD0(a4, 0, (__int64)&v111, v106, v20, 0, v9, 0);
  v77 = v76;
  *(_QWORD *)&v78 = sub_33ED040((_QWORD *)a4, 0x14u);
  *((_QWORD *)&v82 + 1) = v107;
  *(_QWORD *)&v82 = v103;
  *((_QWORD *)&v81 + 1) = v77;
  *(_QWORD *)&v81 = v75;
  *(_QWORD *)a3 = sub_33FC1D0(
                    (_QWORD *)a4,
                    207,
                    (__int64)&v111,
                    *(unsigned __int16 *)(*(_QWORD *)(v103 + 48) + 16LL * (unsigned int)v107),
                    *(_QWORD *)(*(_QWORD *)(v103 + 48) + 16LL * (unsigned int)v107 + 8),
                    v79,
                    v95,
                    v81,
                    v82,
                    v99,
                    v78);
  *(_DWORD *)(a3 + 8) = v80;
  result = 1;
LABEL_8:
  if ( v111 )
  {
    v105 = result;
    sub_B91220((__int64)&v111, v111);
    return v105;
  }
  return result;
}
