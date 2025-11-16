// Function: sub_3777A10
// Address: 0x3777a10
//
void __fastcall sub_3777A10(__int64 a1, __int64 a2, unsigned __int8 **a3, unsigned __int8 **a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  bool v13; // zf
  int v14; // r13d
  _QWORD *v15; // rsi
  unsigned __int16 *v16; // rax
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  unsigned __int64 v19; // xmm0_8
  _QWORD *v20; // rdi
  unsigned int v21; // r12d
  unsigned __int16 *v22; // rax
  unsigned __int8 *v23; // rax
  unsigned __int8 **v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rdx
  __m128i v27; // xmm5
  __int64 v28; // rdx
  __m128i v29; // xmm6
  _QWORD *v30; // rdi
  unsigned __int16 *v31; // rax
  unsigned __int8 *v32; // rax
  unsigned __int8 **v33; // rbx
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned int v36; // r15d
  unsigned __int16 *v37; // rax
  unsigned __int8 *v38; // rax
  unsigned __int8 **v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rdx
  unsigned __int16 *v42; // rax
  unsigned __int8 *v43; // rax
  unsigned __int8 **v44; // rbx
  __int64 v45; // rdx
  unsigned __int64 v46; // [rsp+8h] [rbp-148h]
  unsigned __int64 v47; // [rsp+10h] [rbp-140h]
  unsigned __int64 v48; // [rsp+18h] [rbp-138h]
  __int64 v49; // [rsp+20h] [rbp-130h]
  __int64 v50; // [rsp+28h] [rbp-128h]
  __int64 v51; // [rsp+30h] [rbp-120h]
  unsigned __int8 **v52; // [rsp+38h] [rbp-118h]
  unsigned __int8 **v53; // [rsp+40h] [rbp-110h]
  unsigned int v54; // [rsp+4Ch] [rbp-104h]
  unsigned __int8 *v55; // [rsp+50h] [rbp-100h]
  __int64 v56; // [rsp+58h] [rbp-F8h]
  unsigned __int8 *v57; // [rsp+60h] [rbp-F0h]
  __int64 v58; // [rsp+68h] [rbp-E8h]
  unsigned __int8 *v59; // [rsp+70h] [rbp-E0h]
  __int64 v60; // [rsp+78h] [rbp-D8h]
  unsigned __int8 *v61; // [rsp+80h] [rbp-D0h]
  __int64 v62; // [rsp+88h] [rbp-C8h]
  __m128i v63; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v64; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v65; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i v66; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v67; // [rsp+D0h] [rbp-80h] BYREF
  int v68; // [rsp+D8h] [rbp-78h]
  __m128i v69; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v70; // [rsp+F0h] [rbp-60h]
  __int64 v71; // [rsp+100h] [rbp-50h]
  unsigned __int64 v72; // [rsp+108h] [rbp-48h]
  __int64 v73; // [rsp+110h] [rbp-40h]
  unsigned __int64 v74; // [rsp+118h] [rbp-38h]

  v7 = *(unsigned __int64 **)(a2 + 40);
  v63.m128i_i32[2] = 0;
  v64.m128i_i32[2] = 0;
  v53 = a3;
  v8 = v7[1];
  v63.m128i_i64[0] = 0;
  v64.m128i_i64[0] = 0;
  v9 = *v7;
  v52 = a4;
  sub_375E8D0(a1, v9, v8, (__int64)&v63, (__int64)&v64);
  v10 = *(_QWORD *)(a2 + 40);
  v65.m128i_i32[2] = 0;
  v66.m128i_i32[2] = 0;
  v11 = *(_QWORD *)(v10 + 48);
  v65.m128i_i64[0] = 0;
  v66.m128i_i64[0] = 0;
  sub_375E8D0(a1, *(_QWORD *)(v10 + 40), v11, (__int64)&v65, (__int64)&v66);
  v12 = *(_QWORD *)(a2 + 80);
  v67 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v67, v12, 1);
  v13 = *(_DWORD *)(a2 + 64) == 2;
  v14 = *(_DWORD *)(a2 + 28);
  v68 = *(_DWORD *)(a2 + 72);
  v54 = *(_DWORD *)(a2 + 24);
  if ( v13 )
  {
    v36 = v54;
    v37 = (unsigned __int16 *)(*(_QWORD *)(v63.m128i_i64[0] + 48) + 16LL * v63.m128i_u32[2]);
    v38 = sub_3405C90(
            *(_QWORD **)(a1 + 8),
            v54,
            (__int64)&v67,
            *v37,
            *((_QWORD *)v37 + 1),
            v14,
            a5,
            *(_OWORD *)&v63,
            *(_OWORD *)&v65);
    v39 = v53;
    v62 = v40;
    v61 = v38;
    v41 = v64.m128i_i64[0];
    *v53 = v38;
    *((_DWORD *)v39 + 2) = v62;
    v42 = (unsigned __int16 *)(*(_QWORD *)(v41 + 48) + 16LL * v64.m128i_u32[2]);
    v43 = sub_3405C90(
            *(_QWORD **)(a1 + 8),
            v36,
            (__int64)&v67,
            *v42,
            *((_QWORD *)v42 + 1),
            v14,
            a5,
            *(_OWORD *)&v64,
            *(_OWORD *)&v66);
    v44 = v52;
    v34 = v67;
    v59 = v43;
    v60 = v45;
    *v52 = v43;
    *((_DWORD *)v44 + 2) = v60;
    if ( !v34 )
      return;
  }
  else
  {
    sub_3777990(
      &v69,
      (__int64 *)a1,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
      a5);
    v15 = *(_QWORD **)(a1 + 8);
    v51 = v69.m128i_i64[0];
    v16 = *(unsigned __int16 **)(a2 + 48);
    v50 = v70.m128i_i64[0];
    v47 = _mm_cvtsi32_si128(v70.m128i_u32[2]).m128i_u64[0];
    v46 = _mm_cvtsi32_si128(v69.m128i_u32[2]).m128i_u64[0];
    sub_3408380(
      &v69,
      v15,
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 128LL),
      *v16,
      *((_QWORD *)v16 + 1),
      a5,
      (__int64)&v67);
    v17 = _mm_loadu_si128(&v63);
    v18 = _mm_loadu_si128(&v65);
    v74 = v69.m128i_u32[2];
    v73 = v69.m128i_i64[0];
    v69 = v17;
    v19 = _mm_cvtsi32_si128(v70.m128i_u32[2]).m128i_u64[0];
    v20 = *(_QWORD **)(a1 + 8);
    v71 = v51;
    v21 = v54;
    v49 = v70.m128i_i64[0];
    v70 = v18;
    v72 = v46;
    v48 = v19;
    v22 = (unsigned __int16 *)(*(_QWORD *)(v63.m128i_i64[0] + 48) + 16LL * v63.m128i_u32[2]);
    v23 = sub_33FBA10(v20, v54, (__int64)&v67, *v22, *((_QWORD *)v22 + 1), v14, (__int64)&v69, 4);
    v24 = v53;
    v25 = v49;
    v58 = v26;
    v27 = _mm_loadu_si128(&v64);
    v57 = v23;
    v28 = v64.m128i_i64[0];
    *v53 = v23;
    v29 = _mm_loadu_si128(&v66);
    v73 = v25;
    *((_DWORD *)v24 + 2) = v58;
    v30 = *(_QWORD **)(a1 + 8);
    v69 = v27;
    v70 = v29;
    v71 = v50;
    v72 = v47;
    v74 = v48;
    v31 = (unsigned __int16 *)(*(_QWORD *)(v28 + 48) + 16LL * v64.m128i_u32[2]);
    v32 = sub_33FBA10(v30, v21, (__int64)&v67, *v31, *((_QWORD *)v31 + 1), v14, (__int64)&v69, 4);
    v33 = v52;
    v34 = v67;
    v56 = v35;
    v55 = v32;
    *v52 = v32;
    *((_DWORD *)v33 + 2) = v56;
    if ( !v34 )
      return;
  }
  sub_B91220((__int64)&v67, v34);
}
