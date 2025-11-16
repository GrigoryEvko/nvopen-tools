// Function: sub_3777E20
// Address: 0x3777e20
//
void __fastcall sub_3777E20(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  bool v15; // zf
  unsigned int v16; // r13d
  _QWORD *v17; // rsi
  unsigned __int16 *v18; // rax
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  unsigned __int64 v22; // xmm0_8
  _QWORD *v23; // rdi
  unsigned int v24; // r12d
  unsigned __int16 *v25; // rax
  unsigned __int8 *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdx
  __m128i v30; // xmm6
  __int64 v31; // rdx
  __m128i v32; // xmm7
  __m128i v33; // xmm3
  _QWORD *v34; // rdi
  unsigned __int16 *v35; // rax
  unsigned __int8 *v36; // rax
  _QWORD *v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned int v40; // r15d
  unsigned __int16 *v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rdx
  unsigned __int16 *v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rbx
  __int64 v49; // rdx
  unsigned __int64 v50; // [rsp+8h] [rbp-178h]
  unsigned __int64 v51; // [rsp+10h] [rbp-170h]
  unsigned __int64 v52; // [rsp+18h] [rbp-168h]
  __int64 v53; // [rsp+20h] [rbp-160h]
  __int64 v54; // [rsp+28h] [rbp-158h]
  __int64 v55; // [rsp+30h] [rbp-150h]
  _QWORD *v56; // [rsp+38h] [rbp-148h]
  _QWORD *v57; // [rsp+40h] [rbp-140h]
  unsigned int v58; // [rsp+4Ch] [rbp-134h]
  unsigned __int8 *v59; // [rsp+50h] [rbp-130h]
  __int64 v60; // [rsp+58h] [rbp-128h]
  unsigned __int8 *v61; // [rsp+60h] [rbp-120h]
  __int64 v62; // [rsp+68h] [rbp-118h]
  __int64 v63; // [rsp+70h] [rbp-110h]
  __int64 v64; // [rsp+78h] [rbp-108h]
  __int64 v65; // [rsp+80h] [rbp-100h]
  __int64 v66; // [rsp+88h] [rbp-F8h]
  __m128i v67; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v68; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v69; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v70; // [rsp+C0h] [rbp-C0h] BYREF
  __m128i v71; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v72; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v73; // [rsp+F0h] [rbp-90h] BYREF
  int v74; // [rsp+F8h] [rbp-88h]
  __m128i v75; // [rsp+100h] [rbp-80h] BYREF
  __m128i v76; // [rsp+110h] [rbp-70h]
  __m128i v77; // [rsp+120h] [rbp-60h]
  __int64 v78; // [rsp+130h] [rbp-50h]
  unsigned __int64 v79; // [rsp+138h] [rbp-48h]
  __int64 v80; // [rsp+140h] [rbp-40h]
  unsigned __int64 v81; // [rsp+148h] [rbp-38h]

  v7 = *(unsigned __int64 **)(a2 + 40);
  v67.m128i_i32[2] = 0;
  v68.m128i_i32[2] = 0;
  v57 = a3;
  v8 = v7[1];
  v67.m128i_i64[0] = 0;
  v68.m128i_i64[0] = 0;
  v9 = *v7;
  v56 = a4;
  sub_375E8D0(a1, v9, v8, (__int64)&v67, (__int64)&v68);
  v10 = *(_QWORD *)(a2 + 40);
  v69.m128i_i32[2] = 0;
  v70.m128i_i32[2] = 0;
  v11 = *(_QWORD *)(v10 + 48);
  v69.m128i_i64[0] = 0;
  v70.m128i_i64[0] = 0;
  sub_375E8D0(a1, *(_QWORD *)(v10 + 40), v11, (__int64)&v69, (__int64)&v70);
  v12 = *(_QWORD *)(a2 + 40);
  v71.m128i_i32[2] = 0;
  v72.m128i_i32[2] = 0;
  v13 = *(_QWORD *)(v12 + 88);
  v71.m128i_i64[0] = 0;
  v72.m128i_i64[0] = 0;
  sub_375E8D0(a1, *(_QWORD *)(v12 + 80), v13, (__int64)&v71, (__int64)&v72);
  v14 = *(_QWORD *)(a2 + 80);
  v73 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v73, v14, 1);
  v15 = *(_DWORD *)(a2 + 64) == 3;
  v16 = *(_DWORD *)(a2 + 28);
  v74 = *(_DWORD *)(a2 + 72);
  v58 = *(_DWORD *)(a2 + 24);
  if ( v15 )
  {
    v40 = v58;
    v41 = (unsigned __int16 *)(*(_QWORD *)(v67.m128i_i64[0] + 48) + 16LL * v67.m128i_u32[2]);
    v42 = sub_340EC60(
            *(_QWORD **)(a1 + 8),
            v58,
            (__int64)&v73,
            *v41,
            *((_QWORD *)v41 + 1),
            v16,
            v67.m128i_i64[0],
            v67.m128i_i64[1],
            *(_OWORD *)&v69,
            *(_OWORD *)&v71);
    v43 = v57;
    v66 = v44;
    v65 = v42;
    v45 = v68.m128i_i64[0];
    *v57 = v42;
    *((_DWORD *)v43 + 2) = v66;
    v46 = (unsigned __int16 *)(*(_QWORD *)(v45 + 48) + 16LL * v68.m128i_u32[2]);
    v47 = sub_340EC60(
            *(_QWORD **)(a1 + 8),
            v40,
            (__int64)&v73,
            *v46,
            *((_QWORD *)v46 + 1),
            v16,
            v68.m128i_i64[0],
            v68.m128i_i64[1],
            *(_OWORD *)&v70,
            *(_OWORD *)&v72);
    v48 = v56;
    v38 = v73;
    v63 = v47;
    v64 = v49;
    *v56 = v47;
    *((_DWORD *)v48 + 2) = v64;
    if ( !v38 )
      return;
  }
  else
  {
    sub_3777990(
      &v75,
      (__int64 *)a1,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 128LL),
      a5);
    v17 = *(_QWORD **)(a1 + 8);
    v55 = v75.m128i_i64[0];
    v18 = *(unsigned __int16 **)(a2 + 48);
    v54 = v76.m128i_i64[0];
    v51 = _mm_cvtsi32_si128(v76.m128i_u32[2]).m128i_u64[0];
    v50 = _mm_cvtsi32_si128(v75.m128i_u32[2]).m128i_u64[0];
    sub_3408380(
      &v75,
      v17,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL),
      *v18,
      *((_QWORD *)v18 + 1),
      a5,
      (__int64)&v73);
    v19 = _mm_loadu_si128(&v67);
    v81 = v75.m128i_u32[2];
    v20 = _mm_loadu_si128(&v69);
    v21 = _mm_loadu_si128(&v71);
    v80 = v75.m128i_i64[0];
    v75 = v19;
    v22 = _mm_cvtsi32_si128(v76.m128i_u32[2]).m128i_u64[0];
    v77 = v21;
    v23 = *(_QWORD **)(a1 + 8);
    v24 = v58;
    v53 = v76.m128i_i64[0];
    v78 = v55;
    v76 = v20;
    v79 = v50;
    v25 = (unsigned __int16 *)(*(_QWORD *)(v67.m128i_i64[0] + 48) + 16LL * v67.m128i_u32[2]);
    v52 = v22;
    v26 = sub_33FBA10(v23, v58, (__int64)&v73, *v25, *((_QWORD *)v25 + 1), v16, (__int64)&v75, 5);
    v27 = v57;
    v28 = v53;
    v62 = v29;
    v30 = _mm_loadu_si128(&v68);
    v61 = v26;
    v31 = v68.m128i_i64[0];
    *v57 = v26;
    v32 = _mm_loadu_si128(&v70);
    v80 = v28;
    *((_DWORD *)v27 + 2) = v62;
    v33 = _mm_loadu_si128(&v72);
    v34 = *(_QWORD **)(a1 + 8);
    v75 = v30;
    v78 = v54;
    v76 = v32;
    v77 = v33;
    v79 = v51;
    v81 = v52;
    v35 = (unsigned __int16 *)(*(_QWORD *)(v31 + 48) + 16LL * v68.m128i_u32[2]);
    v36 = sub_33FBA10(v34, v24, (__int64)&v73, *v35, *((_QWORD *)v35 + 1), v16, (__int64)&v75, 5);
    v37 = v56;
    v38 = v73;
    v60 = v39;
    v59 = v36;
    *v56 = v36;
    *((_DWORD *)v37 + 2) = v60;
    if ( !v38 )
      return;
  }
  sub_B91220((__int64)&v73, v38);
}
