// Function: sub_3812C60
// Address: 0x3812c60
//
__int64 __fastcall sub_3812C60(__int64 *a1, __int64 a2)
{
  const __m128i *v3; // rax
  __int64 v4; // rsi
  __m128i v5; // xmm0
  __int64 v6; // rcx
  unsigned __int32 v7; // r14d
  unsigned __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int16 v13; // r14
  __int64 v14; // r8
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int, __int64); // rax
  __int16 v16; // cx
  __int64 v17; // r8
  unsigned int v18; // edx
  int v19; // r9d
  unsigned int v20; // edx
  unsigned __int64 v21; // r13
  __int64 v22; // rsi
  unsigned int v23; // eax
  unsigned int v24; // edx
  int v25; // r9d
  unsigned __int8 *v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // edx
  unsigned __int8 *v29; // r12
  __int16 *v30; // rax
  _QWORD *v31; // rbx
  unsigned __int16 v32; // r9
  __int64 v33; // r8
  unsigned __int64 v34; // r13
  __int128 v35; // rax
  __int64 v36; // r9
  __int64 v37; // r12
  __int64 v39; // rdx
  __int128 v40; // [rsp-20h] [rbp-F0h]
  unsigned int v41; // [rsp+4h] [rbp-CCh]
  __int64 v42; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  __int16 v45; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v46; // [rsp+10h] [rbp-C0h]
  __int64 v47; // [rsp+10h] [rbp-C0h]
  int v48; // [rsp+18h] [rbp-B8h]
  unsigned int v49; // [rsp+18h] [rbp-B8h]
  unsigned int v50; // [rsp+18h] [rbp-B8h]
  __int128 v51; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v52; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+70h] [rbp-60h] BYREF
  int v54; // [rsp+78h] [rbp-58h]
  __int64 v55; // [rsp+80h] [rbp-50h] BYREF
  int v56; // [rsp+88h] [rbp-48h]
  __int64 v57; // [rsp+90h] [rbp-40h]

  v3 = *(const __m128i **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = _mm_loadu_si128(v3);
  v6 = v3->m128i_i64[0];
  v53 = v4;
  v7 = v3->m128i_u32[2];
  v8 = v3[2].m128i_u64[1];
  v9 = v3[3].m128i_i64[0];
  v41 = *(_DWORD *)(v3[5].m128i_i64[0] + 96);
  if ( v4 )
  {
    v44 = v6;
    sub_B96E90((__int64)&v53, v4, 1);
    v6 = v44;
  }
  v10 = *a1;
  v11 = a1[1];
  v54 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(v6 + 48) + 16LL * v7;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
  if ( v15 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v55, v10, *(_QWORD *)(v11 + 64), v13, v14);
    v16 = v56;
    v17 = v57;
  }
  else
  {
    v48 = v15(v10, *(_QWORD *)(v11 + 64), v13, v14);
    v17 = v39;
    v16 = v48;
  }
  v42 = v17;
  v45 = v16;
  sub_380F170((__int64)a1, v5.m128i_u64[0], v5.m128i_i64[1]);
  v52 = v18 | v5.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  sub_380F170((__int64)a1, v8, v9);
  v21 = v20 | v9 & 0xFFFFFFFF00000000LL;
  if ( v13 == 11 )
  {
    v22 = 236;
  }
  else if ( v45 == 11 )
  {
    v22 = 237;
  }
  else if ( v13 == 10 )
  {
    v22 = 240;
  }
  else
  {
    if ( v45 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v22 = 241;
  }
  HIWORD(v23) = HIWORD(v48);
  LOWORD(v23) = v45;
  v49 = v23;
  *(_QWORD *)&v51 = sub_33FAF80(a1[1], v22, (__int64)&v53, v23, v42, v19, v5);
  *((_QWORD *)&v51 + 1) = v24 | v52 & 0xFFFFFFFF00000000LL;
  v26 = sub_33FAF80(a1[1], (unsigned int)v22, (__int64)&v53, v49, v42, v25, v5);
  v27 = *(_QWORD *)(a2 + 80);
  v29 = v26;
  v30 = *(__int16 **)(a2 + 48);
  v31 = (_QWORD *)a1[1];
  v32 = *v30;
  v33 = *((_QWORD *)v30 + 1);
  v34 = v28 | v21 & 0xFFFFFFFF00000000LL;
  v55 = v27;
  if ( v27 )
  {
    v43 = v33;
    v46 = v32;
    sub_B96E90((__int64)&v55, v27, 1);
    v33 = v43;
    v32 = v46;
  }
  v47 = v33;
  v50 = v32;
  v56 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v35 = sub_33ED040(v31, v41);
  *((_QWORD *)&v40 + 1) = v34;
  *(_QWORD *)&v40 = v29;
  v37 = sub_340F900(v31, 0xD0u, (__int64)&v55, v50, v47, v36, v51, v40, v35);
  if ( v55 )
    sub_B91220((__int64)&v55, v55);
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v37;
}
