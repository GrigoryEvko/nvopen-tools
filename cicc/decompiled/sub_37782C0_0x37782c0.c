// Function: sub_37782C0
// Address: 0x37782c0
//
void __fastcall sub_37782C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 *v10; // roff
  __m128i v11; // xmm0
  __int64 v12; // rcx
  __m128i v13; // xmm1
  unsigned __int16 *v14; // rax
  __int64 v15; // r8
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int16 v18; // dx
  __int64 v19; // rax
  __m128i v20; // xmm3
  _QWORD *v21; // r15
  __int64 v22; // rax
  __int16 v23; // dx
  __int64 v24; // rax
  __int64 v25; // rsi
  __m128i v26; // xmm5
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rdx
  unsigned __int16 v34; // cx
  bool v35; // di
  __int64 v36; // rsi
  unsigned int v37; // esi
  unsigned int v38; // r13d
  unsigned __int16 v39; // ax
  __int64 v40; // r9
  __int64 v41; // r8
  unsigned int v42; // r13d
  int v43; // edx
  __int64 v44; // r9
  unsigned __int8 *v45; // rax
  __int64 v46; // rsi
  int v47; // edx
  __int64 v48; // rdx
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  unsigned __int16 v51; // [rsp+10h] [rbp-130h]
  __int64 *v52; // [rsp+18h] [rbp-128h]
  __int64 v53; // [rsp+18h] [rbp-128h]
  __int64 v56; // [rsp+50h] [rbp-F0h] BYREF
  int v57; // [rsp+58h] [rbp-E8h]
  __m128i v58; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v59; // [rsp+70h] [rbp-D0h] BYREF
  __int128 v60; // [rsp+80h] [rbp-C0h] BYREF
  __int128 v61; // [rsp+90h] [rbp-B0h] BYREF
  __int128 v62; // [rsp+A0h] [rbp-A0h] BYREF
  __int128 v63; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v64; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v65; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v66; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v67; // [rsp+E8h] [rbp-58h]
  __m128i v68; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v69; // [rsp+100h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(__int64 **)(a1[1] + 64);
  v56 = v6;
  v52 = v7;
  if ( v6 )
  {
    sub_B96E90((__int64)&v56, v6, 1);
    v8 = *(_QWORD *)(a1[1] + 64);
  }
  else
  {
    v8 = (__int64)v7;
  }
  v9 = *a1;
  v57 = *(_DWORD *)(a2 + 72);
  v10 = *(__int64 **)(a2 + 40);
  v11 = _mm_loadu_si128((const __m128i *)v10);
  v12 = *v10;
  DWORD2(v60) = 0;
  DWORD2(v61) = 0;
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 5));
  v58 = v11;
  DWORD2(v62) = 0;
  v14 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v11.m128i_u32[2]);
  DWORD2(v63) = 0;
  v59 = v13;
  v15 = *((_QWORD *)v14 + 1);
  *(_QWORD *)&v60 = 0;
  *(_QWORD *)&v61 = 0;
  *(_QWORD *)&v62 = 0;
  *(_QWORD *)&v63 = 0;
  sub_2FE6CC0((__int64)&v68, v9, v8, *v14, v15);
  if ( v68.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v58.m128i_u64[0], v58.m128i_i64[1], (__int64)&v60, (__int64)&v61);
    v25 = v59.m128i_i64[0];
    sub_375E8D0((__int64)a1, v59.m128i_u64[0], v59.m128i_i64[1], (__int64)&v62, (__int64)&v63);
  }
  else
  {
    v64.m128i_i64[1] = 0;
    v16 = (_QWORD *)a1[1];
    v64.m128i_i16[0] = 0;
    v65.m128i_i16[0] = 0;
    v65.m128i_i64[1] = 0;
    v17 = *(_QWORD *)(v58.m128i_i64[0] + 48) + 16LL * v58.m128i_u32[2];
    v18 = *(_WORD *)v17;
    v19 = *(_QWORD *)(v17 + 8);
    LOWORD(v66) = v18;
    v67 = v19;
    sub_33D0340((__int64)&v68, (__int64)v16, &v66);
    v20 = _mm_loadu_si128(&v69);
    v64 = _mm_loadu_si128(&v68);
    v65 = v20;
    sub_3408290(
      (__int64)&v68,
      v16,
      (__int128 *)v58.m128i_i8,
      (__int64)&v56,
      (unsigned int *)&v64,
      (unsigned int *)&v65,
      v11);
    v64.m128i_i16[0] = 0;
    *(_QWORD *)&v60 = v68.m128i_i64[0];
    v21 = (_QWORD *)a1[1];
    v65.m128i_i16[0] = 0;
    DWORD2(v60) = v68.m128i_i32[2];
    v64.m128i_i64[1] = 0;
    *(_QWORD *)&v61 = v69.m128i_i64[0];
    v65.m128i_i64[1] = 0;
    DWORD2(v61) = v69.m128i_i32[2];
    v22 = *(_QWORD *)(v59.m128i_i64[0] + 48) + 16LL * v59.m128i_u32[2];
    v23 = *(_WORD *)v22;
    v24 = *(_QWORD *)(v22 + 8);
    LOWORD(v66) = v23;
    v67 = v24;
    sub_33D0340((__int64)&v68, (__int64)v21, &v66);
    v25 = (__int64)v21;
    v26 = _mm_loadu_si128(&v69);
    v64 = _mm_loadu_si128(&v68);
    v65 = v26;
    sub_3408290(
      (__int64)&v68,
      v21,
      (__int128 *)v59.m128i_i8,
      (__int64)&v56,
      (unsigned int *)&v64,
      (unsigned int *)&v65,
      v11);
    *(_QWORD *)&v62 = v68.m128i_i64[0];
    DWORD2(v62) = v68.m128i_i32[2];
    *(_QWORD *)&v63 = v69.m128i_i64[0];
    DWORD2(v63) = v69.m128i_i32[2];
  }
  v29 = *(_QWORD *)(a2 + 48);
  LOWORD(v30) = *(_WORD *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  v68.m128i_i16[0] = v30;
  v68.m128i_i64[1] = v31;
  if ( (_WORD)v30 )
  {
    v32 = 0;
    v33 = (unsigned __int16)v30 - 1;
    v34 = word_4456580[v33];
LABEL_7:
    v35 = (unsigned __int16)(v30 - 176) <= 0x34u;
    LODWORD(v36) = word_4456340[v33];
    LOBYTE(v30) = v35;
    goto LABEL_8;
  }
  v34 = sub_3009970((__int64)&v68, v25, v31, v27, v28);
  LOWORD(v30) = v68.m128i_i16[0];
  v32 = v49;
  if ( v68.m128i_i16[0] )
  {
    v33 = v68.m128i_u16[0] - 1;
    goto LABEL_7;
  }
  v51 = v34;
  v50 = sub_3007240((__int64)&v68);
  v34 = v51;
  v36 = v50;
  v30 = HIDWORD(v50);
  v66 = v36;
  v35 = v30;
LABEL_8:
  v37 = (unsigned int)v36 >> 1;
  v65.m128i_i8[4] = v30;
  v38 = v34;
  v65.m128i_i32[0] = v37;
  if ( v35 )
  {
    v39 = sub_2D43AD0(v34, v37);
    v41 = 0;
    if ( v39 )
      goto LABEL_10;
  }
  else
  {
    v39 = sub_2D43050(v34, v37);
    v41 = 0;
    if ( v39 )
      goto LABEL_10;
  }
  v39 = sub_3009450(v52, v38, v32, v65.m128i_i64[0], 0, v40);
  v41 = v48;
LABEL_10:
  v42 = v39;
  v53 = v41;
  *(_QWORD *)a3 = sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v56, v39, v41, v40, v60, v62);
  *(_DWORD *)(a3 + 8) = v43;
  v45 = sub_3406EB0((_QWORD *)a1[1], *(_DWORD *)(a2 + 24), (__int64)&v56, v42, v53, v44, v61, v63);
  v46 = v56;
  *(_QWORD *)a4 = v45;
  *(_DWORD *)(a4 + 8) = v47;
  if ( v46 )
    sub_B91220((__int64)&v56, v46);
}
