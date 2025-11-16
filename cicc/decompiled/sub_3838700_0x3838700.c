// Function: sub_3838700
// Address: 0x3838700
//
unsigned __int8 *__fastcall sub_3838700(__int64 a1, __int64 a2)
{
  unsigned int *v3; // rax
  __m128i v4; // xmm0
  unsigned __int8 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r9
  __int64 v9; // rsi
  _QWORD *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rcx
  unsigned int v13; // esi
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  unsigned __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int16 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned int v25; // edx
  unsigned __int8 *v26; // r12
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // r9
  __int64 v30; // rsi
  _QWORD *v31; // rbx
  unsigned __int16 *v32; // r13
  __int64 v33; // r12
  __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned __int8 *v36; // rax
  unsigned int v37; // edx
  unsigned int v38; // edx
  __int128 v39; // [rsp-20h] [rbp-D0h]
  __int128 v40; // [rsp+0h] [rbp-B0h]
  __m128i v41; // [rsp+10h] [rbp-A0h]
  __int64 v42; // [rsp+20h] [rbp-90h]
  __int64 v43; // [rsp+28h] [rbp-88h]
  __int64 v44; // [rsp+28h] [rbp-88h]
  __int64 v45; // [rsp+30h] [rbp-80h]
  unsigned __int64 v46; // [rsp+30h] [rbp-80h]
  __int64 v47; // [rsp+30h] [rbp-80h]
  __int128 v48; // [rsp+30h] [rbp-80h]
  __int64 v49; // [rsp+40h] [rbp-70h]
  unsigned __int8 *v50; // [rsp+48h] [rbp-68h]
  __int64 v51; // [rsp+60h] [rbp-50h] BYREF
  int v52; // [rsp+68h] [rbp-48h]

  v3 = *(unsigned int **)(a2 + 40);
  v4 = _mm_loadu_si128((const __m128i *)(v3 + 10));
  v49 = v3[12];
  v50 = (unsigned __int8 *)*((_QWORD *)v3 + 5);
  if ( *(_DWORD *)(a2 + 24) == 398 )
  {
    v18 = *(_QWORD *)v3;
    v19 = *((_QWORD *)v3 + 1);
    v20 = *(_QWORD *)(*(_QWORD *)v3 + 80LL);
    v41 = _mm_loadu_si128((const __m128i *)v3 + 5);
    v40 = (__int128)_mm_loadu_si128((const __m128i *)(v3 + 30));
    v21 = *(_QWORD *)(*(_QWORD *)v3 + 48LL) + 16LL * v3[2];
    v22 = *(_WORD *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    v51 = v20;
    if ( v20 )
    {
      v42 = v19;
      v46 = v18;
      sub_B96E90((__int64)&v51, v20, 1);
      v19 = v42;
      v18 = v46;
    }
    v47 = v19;
    v52 = *(_DWORD *)(v18 + 72);
    v24 = sub_37AE0F0(a1, v18, v19);
    v26 = sub_3400810(
            *(_QWORD **)(a1 + 8),
            v24,
            v47 & 0xFFFFFFFF00000000LL | v25,
            v41.m128i_i64[0],
            v41.m128i_i64[1],
            (__int64)&v51,
            v4,
            v40,
            v22,
            v23);
    v28 = v27;
    if ( v51 )
      sub_B91220((__int64)&v51, v51);
    *(_QWORD *)&v48 = v26;
    *((_QWORD *)&v48 + 1) = v28;
    sub_2FE6CC0(
      (__int64)&v51,
      *(_QWORD *)a1,
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
      *(unsigned __int16 *)(*((_QWORD *)v50 + 6) + 16 * v49),
      *(_QWORD *)(*((_QWORD *)v50 + 6) + 16 * v49 + 8));
    if ( (_BYTE)v51 == 1 )
    {
      v50 = sub_3838540(
              a1,
              v4.m128i_u64[0],
              v4.m128i_i64[1],
              v41.m128i_i64[0],
              v41.m128i_i64[1],
              v4,
              v41.m128i_i64[0],
              v40);
      v49 = v38;
    }
    v30 = *(_QWORD *)(a2 + 80);
    v31 = *(_QWORD **)(a1 + 8);
    v32 = (unsigned __int16 *)(*((_QWORD *)v26 + 6) + 16LL * (unsigned int)v28);
    v33 = *((_QWORD *)v32 + 1);
    v34 = *v32;
    v51 = v30;
    if ( v30 )
    {
      v44 = v34;
      sub_B96E90((__int64)&v51, v30, 1);
      v34 = v44;
    }
    v35 = *(unsigned int *)(a2 + 24);
    v52 = *(_DWORD *)(a2 + 72);
    v36 = sub_33FC130(
            v31,
            v35,
            (__int64)&v51,
            v34,
            v33,
            v29,
            v48,
            __PAIR128__(v49 | v4.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v50),
            *(_OWORD *)&v41,
            v40);
    v15 = v51;
    v16 = v36;
    if ( v51 )
      goto LABEL_7;
  }
  else
  {
    v5 = sub_37AF270(a1, *(_QWORD *)v3, *((_QWORD *)v3 + 1), v4);
    v7 = v6;
    sub_2FE6CC0(
      (__int64)&v51,
      *(_QWORD *)a1,
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
      *(unsigned __int16 *)(16 * v49 + *((_QWORD *)v50 + 6)),
      *(_QWORD *)(16 * v49 + *((_QWORD *)v50 + 6) + 8));
    if ( (_BYTE)v51 == 1 )
    {
      v50 = sub_37AF270(a1, v4.m128i_u64[0], v4.m128i_i64[1], v4);
      v49 = v37;
    }
    v9 = *(_QWORD *)(a2 + 80);
    v10 = *(_QWORD **)(a1 + 8);
    v11 = *(_QWORD *)(*((_QWORD *)v5 + 6) + 16LL * (unsigned int)v7 + 8);
    v12 = *(unsigned __int16 *)(*((_QWORD *)v5 + 6) + 16LL * (unsigned int)v7);
    v51 = v9;
    if ( v9 )
    {
      v43 = v12;
      v45 = v11;
      sub_B96E90((__int64)&v51, v9, 1);
      v12 = v43;
      v11 = v45;
    }
    v13 = *(_DWORD *)(a2 + 24);
    v52 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v39 + 1) = v7;
    *(_QWORD *)&v39 = v5;
    v14 = sub_3406EB0(
            v10,
            v13,
            (__int64)&v51,
            v12,
            v11,
            v8,
            v39,
            __PAIR128__(v49 | v4.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v50));
    v15 = v51;
    v16 = v14;
    if ( v51 )
LABEL_7:
      sub_B91220((__int64)&v51, v15);
  }
  return v16;
}
