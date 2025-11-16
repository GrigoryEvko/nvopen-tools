// Function: sub_2081050
// Address: 0x2081050
//
__int64 *__fastcall sub_2081050(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  __m128i v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rax
  bool v20; // cc
  _QWORD *v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rcx
  bool v26; // r13
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __m128i v30; // xmm5
  __int64 v31; // rax
  char v32; // di
  __int64 v33; // r10
  int v34; // eax
  __int64 v35; // r10
  unsigned int v36; // edx
  __int64 v37; // rax
  __int32 v38; // eax
  _QWORD *v39; // r13
  __m128i v40; // xmm0
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __m128i v43; // xmm3
  __m128i v44; // xmm4
  __int64 v45; // rax
  int v46; // edx
  int v47; // r8d
  int v48; // r9d
  __int64 *v49; // r14
  __int64 v50; // rdx
  __int64 v51; // r13
  __int64 *result; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rax
  unsigned int v58; // edx
  unsigned __int8 v59; // al
  __int64 v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // r13
  __int64 v63; // rdx
  __int64 v64; // rdi
  __int64 v65; // rax
  unsigned int v66; // edx
  unsigned __int8 v67; // al
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 **v71; // rax
  __int64 v72; // [rsp+8h] [rbp-1B8h]
  __int64 v73; // [rsp+18h] [rbp-1A8h]
  __m128i v74; // [rsp+20h] [rbp-1A0h] BYREF
  __m128i v75; // [rsp+30h] [rbp-190h] BYREF
  unsigned int v76; // [rsp+44h] [rbp-17Ch]
  __int64 v77; // [rsp+48h] [rbp-178h]
  __int64 v78; // [rsp+50h] [rbp-170h]
  __int64 *v79; // [rsp+58h] [rbp-168h]
  __int64 v80; // [rsp+60h] [rbp-160h]
  __int64 *v81; // [rsp+68h] [rbp-158h]
  __int64 *v82; // [rsp+70h] [rbp-150h]
  __int64 v83; // [rsp+78h] [rbp-148h]
  __int64 v84; // [rsp+80h] [rbp-140h]
  __int64 v85; // [rsp+88h] [rbp-138h]
  __int64 *v86; // [rsp+90h] [rbp-130h]
  __int64 v87; // [rsp+98h] [rbp-128h]
  __int64 v88; // [rsp+A0h] [rbp-120h]
  __int64 v89; // [rsp+A8h] [rbp-118h]
  __int64 *v90; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v91; // [rsp+B8h] [rbp-108h] BYREF
  __int64 v92; // [rsp+C0h] [rbp-100h] BYREF
  int v93; // [rsp+C8h] [rbp-F8h]
  unsigned int v94; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v95; // [rsp+D8h] [rbp-E8h]
  __m128i v96; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v97; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v98; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v99; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v100; // [rsp+120h] [rbp-A0h]
  __int128 v101; // [rsp+130h] [rbp-90h] BYREF
  __m128i v102; // [rsp+140h] [rbp-80h]
  __m128i v103; // [rsp+150h] [rbp-70h]
  __m128i v104; // [rsp+160h] [rbp-60h]
  __m128i v105; // [rsp+170h] [rbp-50h]
  __m128i v106; // [rsp+180h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 536);
  v8 = *(_QWORD *)a1;
  v92 = 0;
  v93 = v7;
  if ( v8 )
  {
    v81 = &v92;
    if ( &v92 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v92 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v92, v9, 2);
    }
  }
  else
  {
    v81 = &v92;
  }
  v10 = 3 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v79 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v11 = sub_20685E0(a1, *(__int64 **)(a2 + 8 * v10), a3, a4, a5);
  v75.m128i_i64[1] = v12;
  LODWORD(v12) = *(_DWORD *)(a2 + 20);
  v75.m128i_i64[0] = (__int64)v11;
  v13.m128i_i64[0] = (__int64)sub_20685E0(a1, *(__int64 **)(a2 + 24 * (2 - (v12 & 0xFFFFFFF))), a3, a4, a5);
  v14 = *(_QWORD *)a2;
  v74 = v13;
  v13.m128i_i64[0] = *(_QWORD *)(a1 + 552);
  v15 = *(_QWORD *)(v13.m128i_i64[0] + 16);
  v16 = sub_1E0A0C0(*(_QWORD *)(v13.m128i_i64[0] + 32));
  LOBYTE(v17) = sub_204D4D0(v15, v16, v14);
  v95 = v18;
  LODWORD(v18) = *(_DWORD *)(a2 + 20);
  v94 = v17;
  v19 = *(_QWORD *)(a2 + 24 * (1 - (v18 & 0xFFFFFFF)));
  v20 = *(_DWORD *)(v19 + 32) <= 0x40u;
  v21 = *(_QWORD **)(v19 + 24);
  if ( !v20 )
    v21 = (_QWORD *)*v21;
  LODWORD(v80) = (_DWORD)v21;
  if ( !(_DWORD)v21 )
    LODWORD(v80) = sub_1D172F0(*(_QWORD *)(a1 + 552), v94, v95);
  v99 = 0u;
  v100 = 0;
  sub_14A8180(a2, v99.m128i_i64, 0);
  v22 = *(_QWORD *)(a2 + 48);
  if ( v22 || *(__int16 *)(a2 + 18) < 0 )
    LODWORD(v22) = sub_1625790(a2, 4);
  v23 = *(_QWORD *)(a1 + 552);
  v96.m128i_i64[0] = 0;
  v24 = *(_QWORD *)(v23 + 184);
  v96.m128i_i32[2] = 0;
  v97.m128i_i64[0] = 0;
  v73 = v24;
  v25 = *(_QWORD *)(v23 + 176);
  v97.m128i_i32[2] = 0;
  LODWORD(v23) = *(_DWORD *)(v23 + 184);
  v77 = v25;
  v76 = v23;
  v98.m128i_i64[0] = 0;
  v98.m128i_i32[2] = 0;
  v90 = v79;
  v26 = sub_2080CA0((__int64 *)&v90, (__int64)&v96, v97.m128i_i64, (__int64)&v98, a1, a3, a4, a5);
  if ( v26 )
  {
    if ( *(_QWORD *)(a1 + 568) )
    {
      v27 = *(_QWORD *)(a1 + 552);
      v78 = *(_QWORD *)(a1 + 568);
      v28 = sub_1E0A0C0(*(_QWORD *)(v27 + 32));
      v29 = sub_127FA20(v28, *(_QWORD *)a2);
      v30 = _mm_load_si128(&v99);
      *(_QWORD *)&v101 = v90;
      *((_QWORD *)&v101 + 1) = (unsigned __int64)(v29 + 7) >> 3;
      v102 = v30;
      v103.m128i_i64[0] = v100;
      LOBYTE(v78) = sub_134CBB0(v78, (__int64)&v101, 0);
      if ( (_BYTE)v78 )
      {
        v31 = *(_QWORD *)(a1 + 552);
        v32 = v94;
        v76 = 0;
        v33 = *(_QWORD *)(v31 + 32);
        v77 = v31 + 88;
        if ( !(_BYTE)v94 )
          goto LABEL_15;
LABEL_26:
        v36 = ((unsigned int)sub_2045180(v32) + 7) >> 3;
        if ( v26 )
          goto LABEL_16;
        goto LABEL_27;
      }
    }
  }
  v54 = *(_QWORD *)(a1 + 552);
  v32 = v94;
  LOBYTE(v78) = 0;
  v33 = *(_QWORD *)(v54 + 32);
  if ( (_BYTE)v94 )
    goto LABEL_26;
LABEL_15:
  v72 = v33;
  v34 = sub_1F58D40((__int64)&v94);
  v35 = v72;
  v36 = (unsigned int)(v34 + 7) >> 3;
  if ( v26 )
  {
LABEL_16:
    v102.m128i_i8[0] = 0;
    v101 = (unsigned __int64)v90;
    if ( v90 )
    {
      v37 = *v90;
      if ( *(_BYTE *)(*v90 + 8) == 16 )
        v37 = **(_QWORD **)(v37 + 16);
      v38 = *(_DWORD *)(v37 + 8) >> 8;
    }
    else
    {
      v38 = 0;
    }
    v102.m128i_i32[1] = v38;
    v80 = sub_1E0B8E0(v35, 1u, v36, v80, (int)&v99, v22, v101, v102.m128i_i64[0], 1u, 0, 0);
    goto LABEL_21;
  }
LABEL_27:
  v102.m128i_i64[0] = 0;
  v101 = 0u;
  v55 = sub_1E0B8E0(v35, 1u, v36, v80, (int)&v99, v22, 0, 0, 1u, 0, 0);
  v56 = *(_QWORD *)(a1 + 552);
  v80 = v55;
  v57 = sub_1E0A0C0(*(_QWORD *)(v56 + 32));
  v58 = 8 * sub_15A9520(v57, 0);
  if ( v58 == 32 )
  {
    v59 = 5;
  }
  else if ( v58 > 0x20 )
  {
    v59 = 6;
    if ( v58 != 64 )
    {
      v59 = 7;
      if ( v58 != 128 )
        v59 = v26;
    }
  }
  else
  {
    v59 = 3;
    if ( v58 != 8 )
      v59 = 4 * (v58 == 16);
  }
  v88 = sub_1D38BB0(v56, 0, (__int64)v81, v59, 0, 0, a3, *(double *)a4.m128i_i64, a5, 0);
  v89 = v60;
  v96.m128i_i64[0] = v88;
  v96.m128i_i32[2] = v60;
  v61 = sub_20685E0(a1, v79, a3, a4, a5);
  v62 = *(_QWORD *)(a1 + 552);
  v87 = v63;
  v86 = v61;
  v64 = *(_QWORD *)(v62 + 32);
  v97.m128i_i64[0] = (__int64)v61;
  v97.m128i_i32[2] = v63;
  v65 = sub_1E0A0C0(v64);
  v66 = 8 * sub_15A9520(v65, 0);
  if ( v66 == 32 )
  {
    v67 = 5;
  }
  else if ( v66 > 0x20 )
  {
    v67 = 6;
    if ( v66 != 64 )
    {
      v67 = 0;
      if ( v66 == 128 )
        v67 = 7;
    }
  }
  else
  {
    v67 = 3;
    if ( v66 != 8 )
      v67 = 4 * (v66 == 16);
  }
  v68 = sub_1D38BB0(v62, 1, (__int64)v81, v67, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v85 = v69;
  v84 = v68;
  v98.m128i_i64[0] = v68;
  v98.m128i_i32[2] = v69;
LABEL_21:
  v39 = *(_QWORD **)(a1 + 552);
  v40 = _mm_load_si128(&v75);
  v41 = _mm_load_si128(&v74);
  v42 = _mm_load_si128(&v96);
  v43 = _mm_load_si128(&v97);
  v44 = _mm_load_si128(&v98);
  *(_QWORD *)&v101 = v77;
  v102 = v40;
  v103 = v41;
  v104 = v42;
  v105 = v43;
  v106 = v44;
  *((_QWORD *)&v101 + 1) = v76 | v73 & 0xFFFFFFFF00000000LL;
  v45 = sub_1D252B0((__int64)v39, v94, v95, 1, 0);
  v49 = sub_1D24AE0(v39, v45, v46, v94, v95, (__int64)v81, (__int64 *)&v101, 6, v80);
  v51 = v50;
  if ( !(_BYTE)v78 )
  {
    v70 = *(unsigned int *)(a1 + 112);
    if ( (unsigned int)v70 >= *(_DWORD *)(a1 + 116) )
    {
      sub_16CD150(a1 + 104, (const void *)(a1 + 120), 0, 16, v47, v48);
      v70 = *(unsigned int *)(a1 + 112);
    }
    v71 = (__int64 **)(*(_QWORD *)(a1 + 104) + 16 * v70);
    *v71 = v49;
    v71[1] = (__int64 *)1;
    ++*(_DWORD *)(a1 + 112);
  }
  v91 = a2;
  result = sub_205F5C0(a1 + 8, &v91);
  v83 = v51;
  v53 = v92;
  v82 = v49;
  result[1] = (__int64)v49;
  *((_DWORD *)result + 4) = v83;
  if ( v53 )
    return (__int64 *)sub_161E7C0((__int64)v81, v53);
  return result;
}
