// Function: sub_32F3470
// Address: 0x32f3470
//
__int64 __fastcall sub_32F3470(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 *v5; // r15
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int16 *v10; // rax
  __int64 (__fastcall *v11)(__int64 *, __int64, __int64, __int64, __int64); // r13
  __int16 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // rsi
  int v18; // edx
  int v19; // r13d
  __int64 v20; // rdi
  __m128i v21; // xmm3
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r13
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // rsi
  unsigned int v31; // edx
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // rsi
  unsigned int v35; // ecx
  __int64 v36; // r14
  __int128 v37; // rax
  __int128 v38; // rax
  __int64 v39; // r12
  __int128 v40; // rax
  int v41; // r9d
  __int64 v42; // rax
  __int32 v43; // ecx
  int v44; // ebx
  __int64 v45; // r12
  unsigned int v46; // edx
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int16 v49; // ax
  __int64 v50; // rdx
  __int64 v51; // r13
  int v52; // esi
  __int64 v53; // rax
  int v54; // r9d
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // rdx
  __m128i v59; // xmm4
  __int64 v60; // rcx
  unsigned __int128 v61; // kr00_16
  int v62; // r9d
  __int64 v63; // r14
  __int128 v64; // rax
  int v65; // r9d
  __int64 v66; // r13
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // r9
  bool v72; // al
  unsigned __int16 *v73; // rax
  __int64 v74; // [rsp+0h] [rbp-F0h]
  __int64 v75; // [rsp+0h] [rbp-F0h]
  __int64 v76; // [rsp+20h] [rbp-D0h]
  __int128 v77; // [rsp+20h] [rbp-D0h]
  __int64 v78; // [rsp+30h] [rbp-C0h]
  __int128 v79; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v80; // [rsp+30h] [rbp-C0h]
  unsigned int v81; // [rsp+40h] [rbp-B0h]
  unsigned int v82; // [rsp+44h] [rbp-ACh]
  unsigned __int16 v83; // [rsp+50h] [rbp-A0h]
  __int64 v84; // [rsp+50h] [rbp-A0h]
  __int128 v85; // [rsp+50h] [rbp-A0h]
  __m128i v86; // [rsp+60h] [rbp-90h] BYREF
  __m128i v87; // [rsp+70h] [rbp-80h] BYREF
  __int64 v88; // [rsp+80h] [rbp-70h] BYREF
  __int64 v89; // [rsp+88h] [rbp-68h]
  __int64 v90; // [rsp+90h] [rbp-60h] BYREF
  int v91; // [rsp+98h] [rbp-58h]
  __m128i v92; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v93; // [rsp+B0h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = (__int64 *)a1[1];
  v6 = _mm_loadu_si128((const __m128i *)v4);
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v8 = *(_QWORD *)(v4 + 40);
  LODWORD(v4) = *(_DWORD *)(v4 + 48);
  v9 = *v5;
  v86 = v6;
  v82 = v4;
  v10 = *(__int16 **)(a2 + 48);
  v11 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(v9 + 528);
  v76 = v8;
  v12 = *v10;
  v13 = *((_QWORD *)v10 + 1);
  v87 = v7;
  v89 = v13;
  v14 = *a1;
  LOWORD(v88) = v12;
  v78 = *(_QWORD *)(v14 + 64);
  v15 = sub_2E79000(*(__int64 **)(v14 + 40));
  v16 = v11(v5, v15, v78, v88, v89);
  v17 = *(_QWORD *)(a2 + 80);
  v83 = v16;
  v19 = v18;
  v90 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v90, v17, 1);
  v20 = *a1;
  v21 = _mm_load_si128(&v87);
  v91 = *(_DWORD *)(a2 + 72);
  v92 = _mm_load_si128(&v86);
  v93 = v21;
  v22 = sub_3402EA0(v20, 59, (unsigned int)&v90, v88, v89, 0, (__int64)&v92, 2);
  v25 = v74;
  if ( v22 )
    goto LABEL_4;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
      goto LABEL_10;
  }
  else if ( !sub_30070B0((__int64)&v88) )
  {
    goto LABEL_10;
  }
  v22 = sub_3295970(a1, a2, (__int64)&v90, v25, v24);
  if ( v22 )
    goto LABEL_4;
LABEL_10:
  v28 = sub_33DFBC0(v87.m128i_i64[0], v87.m128i_i64[1], 0, 0);
  v29 = v28;
  if ( !v28 )
    goto LABEL_24;
  v30 = *(_QWORD *)(v28 + 96);
  v31 = *(_DWORD *)(v30 + 32);
  if ( !v31 )
    goto LABEL_33;
  v32 = v30 + 24;
  if ( v31 > 0x40 )
  {
    v81 = *(_DWORD *)(v30 + 32);
    v33 = sub_C445E0(v32);
    v31 = v81;
    v32 = v30 + 24;
    if ( v81 != v33 )
      goto LABEL_14;
LABEL_33:
    v22 = sub_3407430(*a1, v86.m128i_i64[0], v86.m128i_i64[1], &v90, (unsigned int)v88, v89);
    goto LABEL_4;
  }
  if ( *(_QWORD *)(v30 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v31) )
    goto LABEL_33;
LABEL_14:
  v34 = *(_QWORD *)(v30 + 24);
  v35 = v31 - 1;
  if ( v31 <= 0x40 )
  {
    if ( 1LL << v35 == v34 )
      goto LABEL_17;
  }
  else if ( (*(_QWORD *)(v34 + 8LL * (v35 >> 6)) & (1LL << v35)) != 0 && v35 == (unsigned int)sub_C44590(v32) )
  {
LABEL_17:
    v36 = *a1;
    *(_QWORD *)&v37 = sub_3400BD0(*a1, 0, (unsigned int)&v90, v88, v89, 0, 0);
    LODWORD(v75) = 0;
    v77 = v37;
    *(_QWORD *)&v38 = sub_3400BD0(*a1, 1, (unsigned int)&v90, v88, v89, 0, v75);
    v39 = *a1;
    v79 = v38;
    *(_QWORD *)&v40 = sub_33ED040(v39, 17);
    v42 = sub_340F900(v39, 208, (unsigned int)&v90, v83, v19, v41, *(_OWORD *)&v86, *(_OWORD *)&v87, v40);
    v43 = v88;
    v44 = v89;
    v45 = v42;
    v47 = v46;
    v48 = *(_QWORD *)(v42 + 48) + 16LL * v46;
    v49 = *(_WORD *)v48;
    v50 = *(_QWORD *)(v48 + 8);
    v51 = v47;
    v92.m128i_i16[0] = v49;
    v92.m128i_i64[1] = v50;
    if ( v49 )
    {
      v52 = ((unsigned __int16)(v49 - 17) < 0xD4u) + 205;
    }
    else
    {
      v87.m128i_i64[0] = v88;
      v72 = sub_30070B0((__int64)&v92);
      v43 = v87.m128i_i32[0];
      v52 = 205 - (!v72 - 1);
    }
    v26 = sub_340EC60(v36, v52, (unsigned int)&v90, v43, v44, 0, v45, v51, v79, v77);
    goto LABEL_5;
  }
LABEL_24:
  v53 = sub_3269740(a2, *a1);
  v84 = v23;
  v26 = v53;
  if ( v53 )
    goto LABEL_5;
  v22 = sub_329BF20(a1, a2);
  if ( v22 )
  {
LABEL_4:
    v26 = v22;
    goto LABEL_5;
  }
  if ( (unsigned __int8)sub_33DD2A0(*a1, v87.m128i_i64[0], v87.m128i_i64[1], 0)
    && (unsigned __int8)sub_33DD2A0(*a1, v86.m128i_i64[0], v86.m128i_i64[1], 0) )
  {
    v73 = (unsigned __int16 *)(*(_QWORD *)(v76 + 48) + 16LL * v82);
    v26 = sub_3406EB0(*a1, 60, (unsigned int)&v90, *v73, *((_QWORD *)v73 + 1), v54, *(_OWORD *)&v86, *(_OWORD *)&v87);
  }
  else
  {
    v55 = sub_32CEA50(a1, v86.m128i_i64[0], v86.m128i_i64[1], v87.m128i_i64[0], v87.m128i_i64[1], a2);
    if ( v55 )
    {
      *((_QWORD *)&v85 + 1) = v56;
      v57 = *a1;
      v58 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)&v85 = v55;
      v59 = _mm_load_si128(&v86);
      v60 = *(unsigned int *)(a2 + 68);
      v93 = _mm_load_si128(&v87);
      v92 = v59;
      v61 = v85;
      v63 = sub_33D01C0(v57, 61, v58, v60, &v92, 2);
      if ( v63 )
      {
        v80 = v85;
        *(_QWORD *)&v64 = sub_3406EB0(*a1, 58, (unsigned int)&v90, v88, v89, v62, v85, *(_OWORD *)&v87);
        *(_QWORD *)&v85 = v64;
        v66 = sub_3406EB0(*a1, 57, (unsigned int)&v90, v88, v89, v65, *(_OWORD *)&v86, v64);
        v87.m128i_i64[0] = v67;
        sub_32B3E80((__int64)a1, v85, 1, 0, v68, v69);
        sub_32B3E80((__int64)a1, v66, 1, 0, v70, v71);
        v92.m128i_i64[0] = v66;
        v92.m128i_i64[1] = v87.m128i_i64[0];
        sub_32EB790((__int64)a1, v63, v92.m128i_i64, 1, 1);
        v61 = __PAIR128__(*((unsigned __int64 *)&v85 + 1), v80);
      }
      v23 = v61 >> 64;
      v26 = v61;
    }
    else if ( v29
           && !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)a1[1] + 200LL))(
                 a1[1],
                 **(unsigned __int16 **)(a2 + 48),
                 *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                 *(_QWORD *)(**(_QWORD **)(*a1 + 40) + 120LL))
           || (v26 = sub_32EC020(a1, a2)) == 0 )
    {
      v26 = 0;
      v23 = v84 & 0xFFFFFFFF00000000LL;
    }
  }
LABEL_5:
  if ( v90 )
  {
    v87.m128i_i64[0] = v23;
    sub_B91220((__int64)&v90, v90);
  }
  return v26;
}
