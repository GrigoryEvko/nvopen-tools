// Function: sub_2081870
// Address: 0x2081870
//
__int64 *__fastcall sub_2081870(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r13d
  __int64 *v15; // r15
  __m128i v16; // rax
  bool v17; // cc
  _QWORD *v18; // rax
  bool v19; // r13
  __int64 *v20; // rcx
  __int64 v21; // rsi
  int v22; // edx
  __int64 *v23; // rcx
  __int64 v24; // r11
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rax
  __int64 v28; // r15
  __int128 v29; // rax
  _QWORD *v30; // r13
  __m128i v31; // xmm0
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __int64 v35; // rcx
  __int64 v36; // rax
  int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // r13
  __int64 v42; // r15
  __int64 *v43; // r8
  __int64 *result; // rax
  __int64 v45; // rsi
  int v46; // eax
  __int64 v47; // rdi
  __int64 v48; // rax
  unsigned int v49; // edx
  unsigned __int8 v50; // al
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rax
  unsigned int v57; // edx
  unsigned __int8 v58; // al
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // [rsp+8h] [rbp-1B8h]
  __int64 *v62; // [rsp+10h] [rbp-1B0h]
  __int64 v63; // [rsp+18h] [rbp-1A8h]
  __m128i v64; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v65; // [rsp+30h] [rbp-190h]
  __int64 *v66; // [rsp+38h] [rbp-188h]
  __int64 *v67; // [rsp+40h] [rbp-180h]
  __int64 v68; // [rsp+48h] [rbp-178h]
  __int64 v69; // [rsp+50h] [rbp-170h]
  __int64 v70; // [rsp+58h] [rbp-168h]
  __int64 v71; // [rsp+60h] [rbp-160h]
  __int64 v72; // [rsp+68h] [rbp-158h]
  __int64 v73; // [rsp+70h] [rbp-150h]
  __int64 v74; // [rsp+78h] [rbp-148h]
  __int64 v75; // [rsp+80h] [rbp-140h]
  __int64 v76; // [rsp+88h] [rbp-138h]
  __int64 *v77; // [rsp+90h] [rbp-130h]
  __int64 v78; // [rsp+98h] [rbp-128h]
  __int64 v79; // [rsp+A0h] [rbp-120h]
  __int64 v80; // [rsp+A8h] [rbp-118h]
  __int64 *v81; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v82; // [rsp+B8h] [rbp-108h] BYREF
  __int64 v83; // [rsp+C0h] [rbp-100h] BYREF
  int v84; // [rsp+C8h] [rbp-F8h]
  unsigned int v85; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v86; // [rsp+D8h] [rbp-E8h]
  __m128i v87; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v88; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v89; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v90[4]; // [rsp+110h] [rbp-B0h] BYREF
  __int128 v91; // [rsp+130h] [rbp-90h] BYREF
  __int64 v92; // [rsp+140h] [rbp-80h]
  __int64 v93; // [rsp+148h] [rbp-78h]
  __m128i v94; // [rsp+150h] [rbp-70h]
  __m128i v95; // [rsp+160h] [rbp-60h]
  __m128i v96; // [rsp+170h] [rbp-50h]
  __m128i v97; // [rsp+180h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 536);
  v8 = *(_QWORD *)a1;
  v83 = 0;
  v84 = v7;
  if ( v8 )
  {
    if ( &v83 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v83 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v83, v9, 2);
    }
  }
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(__int64 **)(a2 - 24 * v10);
  v66 = *(__int64 **)(a2 + 24 * (1 - v10));
  v12 = sub_20685E0(a1, v11, a3, a4, a5);
  v63 = v13;
  v14 = v13;
  LODWORD(v13) = *(_DWORD *)(a2 + 20);
  v15 = v12;
  v65 = (__int64)v12;
  v16.m128i_i64[0] = (__int64)sub_20685E0(a1, *(__int64 **)(a2 + 24 * (3 - (v13 & 0xFFFFFFF))), a3, a4, a5);
  v64 = v16;
  v16.m128i_i64[0] = v15[5] + 16LL * v14;
  v16.m128i_i8[8] = *(_BYTE *)v16.m128i_i64[0];
  v16.m128i_i64[0] = *(_QWORD *)(v16.m128i_i64[0] + 8);
  LOBYTE(v85) = v16.m128i_i8[8];
  v16.m128i_i32[2] = *(_DWORD *)(a2 + 20);
  v86 = v16.m128i_i64[0];
  v16.m128i_i64[0] = *(_QWORD *)(a2 + 24 * (2LL - (v16.m128i_i32[2] & 0xFFFFFFF)));
  v17 = *(_DWORD *)(v16.m128i_i64[0] + 32) <= 0x40u;
  v18 = *(_QWORD **)(v16.m128i_i64[0] + 24);
  if ( !v17 )
    v18 = (_QWORD *)*v18;
  LODWORD(v67) = (_DWORD)v18;
  if ( !(_DWORD)v18 )
    LODWORD(v67) = sub_1D172F0(*(_QWORD *)(a1 + 552), v85, v86);
  memset(v90, 0, 24);
  sub_14A8180(a2, v90, 0);
  v87.m128i_i64[0] = 0;
  v87.m128i_i32[2] = 0;
  v88.m128i_i64[0] = 0;
  v88.m128i_i32[2] = 0;
  v89.m128i_i64[0] = 0;
  v89.m128i_i32[2] = 0;
  v81 = v66;
  v19 = sub_2080CA0((__int64 *)&v81, (__int64)&v87, v88.m128i_i64, (__int64)&v89, a1, a3, a4, a5);
  if ( v19 )
  {
    v20 = v81;
    v21 = (__int64)v81;
  }
  else
  {
    v21 = 0;
    v20 = 0;
  }
  if ( (_BYTE)v85 )
  {
    v22 = sub_2045180(v85);
  }
  else
  {
    v61 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
    v62 = v20;
    v46 = sub_1F58D40((__int64)&v85);
    v24 = v61;
    v23 = v62;
    v22 = v46;
  }
  v91 = (unsigned __int64)v21;
  v25 = 0;
  v26 = (unsigned int)(v22 + 7) >> 3;
  LOBYTE(v92) = 0;
  if ( v23 )
  {
    v27 = *v23;
    if ( *(_BYTE *)(*v23 + 8) == 16 )
      v27 = **(_QWORD **)(v27 + 16);
    v25 = *(_DWORD *)(v27 + 8) >> 8;
  }
  HIDWORD(v92) = v25;
  v28 = sub_1E0B8E0(v24, 2u, v26, (int)v67, (int)v90, 0, v91, v92, 1u, 0, 0);
  if ( !v19 )
  {
    v47 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
    v67 = *(__int64 **)(a1 + 552);
    v48 = sub_1E0A0C0(v47);
    v49 = 8 * sub_15A9520(v48, 0);
    if ( v49 == 32 )
    {
      v50 = 5;
    }
    else if ( v49 > 0x20 )
    {
      v50 = 6;
      if ( v49 != 64 )
      {
        v50 = 7;
        if ( v49 != 128 )
          v50 = 0;
      }
    }
    else
    {
      v50 = 3;
      if ( v49 != 8 )
        v50 = 4 * (v49 == 16);
    }
    v79 = sub_1D38BB0((__int64)v67, 0, (__int64)&v83, v50, 0, 0, a3, *(double *)a4.m128i_i64, a5, 0);
    v80 = v51;
    v87.m128i_i64[0] = v79;
    v87.m128i_i32[2] = v51;
    v52 = sub_20685E0(a1, v66, a3, a4, a5);
    v53 = *(_QWORD *)(a1 + 552);
    v78 = v54;
    v77 = v52;
    v55 = *(_QWORD *)(v53 + 32);
    v88.m128i_i64[0] = (__int64)v52;
    v88.m128i_i32[2] = v54;
    v56 = sub_1E0A0C0(v55);
    v57 = 8 * sub_15A9520(v56, 0);
    if ( v57 == 32 )
    {
      v58 = 5;
    }
    else if ( v57 > 0x20 )
    {
      v58 = 6;
      if ( v57 != 64 )
      {
        v58 = 0;
        if ( v57 == 128 )
          v58 = 7;
      }
    }
    else
    {
      v58 = 3;
      if ( v57 != 8 )
        v58 = 4 * (v57 == 16);
    }
    v59 = sub_1D38BB0(v53, 1, (__int64)&v83, v58, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v76 = v60;
    v75 = v59;
    v89.m128i_i64[0] = v59;
    v89.m128i_i32[2] = v60;
  }
  *(_QWORD *)&v29 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v30 = *(_QWORD **)(a1 + 552);
  v31 = _mm_load_si128(&v64);
  v91 = v29;
  v32 = _mm_loadu_si128(&v87);
  v33 = _mm_loadu_si128(&v88);
  v34 = _mm_loadu_si128(&v89);
  v92 = v65;
  v67 = (__int64 *)&v91;
  v68 = 6;
  v94 = v31;
  v95 = v32;
  v96 = v33;
  v97 = v34;
  v93 = v63;
  v36 = sub_1D29190((__int64)v30, 1u, 0, v35, (__int64)&v91, 6);
  v38 = sub_1D24800(v30, v36, v37, v85, v86, (__int64)&v83, v67, v68, v28);
  v40 = *(_QWORD *)(a1 + 552);
  v41 = (__int64)v38;
  v42 = v39;
  if ( v38 )
  {
    v67 = *(__int64 **)(a1 + 552);
    nullsub_686();
    v43 = v67;
    v74 = v42;
    v73 = v41;
    v67[22] = v41;
    *((_DWORD *)v43 + 46) = v74;
    sub_1D23870();
  }
  else
  {
    v70 = v39;
    v69 = 0;
    *(_QWORD *)(v40 + 176) = 0;
    *(_DWORD *)(v40 + 184) = v70;
  }
  v82 = a2;
  result = sub_205F5C0(a1 + 8, &v82);
  v72 = v42;
  v45 = v83;
  v71 = v41;
  result[1] = v41;
  *((_DWORD *)result + 4) = v72;
  if ( v45 )
    return (__int64 *)sub_161E7C0((__int64)&v83, v45);
  return result;
}
