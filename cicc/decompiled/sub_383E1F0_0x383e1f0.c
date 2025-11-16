// Function: sub_383E1F0
// Address: 0x383e1f0
//
unsigned __int8 *__fastcall sub_383E1F0(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rbx
  __int128 v11; // rax
  __int64 v12; // r9
  _QWORD *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // rcx
  unsigned int v17; // esi
  unsigned __int8 *v18; // rax
  __int64 v19; // rsi
  unsigned __int8 *v20; // r12
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __int64 v24; // r9
  _QWORD *v25; // r12
  __int64 v26; // rsi
  __int64 v27; // r13
  __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // rax
  unsigned int v32; // edx
  unsigned __int8 *v33; // rax
  unsigned int v34; // edx
  __int128 v35; // [rsp-30h] [rbp-D0h]
  __int128 v36; // [rsp-10h] [rbp-B0h]
  __int64 v37; // [rsp+0h] [rbp-A0h]
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  __int128 v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int128 v42; // [rsp+20h] [rbp-80h]
  __int128 v43; // [rsp+30h] [rbp-70h]
  __int64 v44; // [rsp+40h] [rbp-60h]
  unsigned __int64 v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+50h] [rbp-50h] BYREF
  int v47; // [rsp+58h] [rbp-48h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(v7 + 40);
  v9 = *(_QWORD *)(v7 + 48);
  v44 = *(unsigned int *)(v7 + 48);
  v10 = 16 * v44;
  v45 = v8;
  if ( *(_DWORD *)(a2 + 24) == 397 )
  {
    v22 = _mm_loadu_si128((const __m128i *)(v7 + 80));
    v42 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 120));
    *(_QWORD *)&v40 = sub_382E5B0(
                        (__int64)a1,
                        *(_QWORD *)v7,
                        *(_QWORD *)(v7 + 8),
                        v22.m128i_i64[0],
                        v22.m128i_i64[1],
                        v22,
                        a7,
                        v42);
    *((_QWORD *)&v40 + 1) = v23;
    sub_2FE6CC0(
      (__int64)&v46,
      *a1,
      *(_QWORD *)(a1[1] + 64),
      *(unsigned __int16 *)(v10 + *(_QWORD *)(v8 + 48)),
      *(_QWORD *)(v10 + *(_QWORD *)(v8 + 48) + 8));
    v24 = a2;
    if ( (_BYTE)v46 == 1 )
    {
      v33 = sub_3838540((__int64)a1, v8, v9, v22.m128i_i64[0], v22.m128i_i64[1], v22, a2, v42);
      v24 = a2;
      v45 = (unsigned __int64)v33;
      v44 = v34;
    }
    v25 = (_QWORD *)a1[1];
    v26 = *(_QWORD *)(v24 + 80);
    v27 = *(_QWORD *)(*(_QWORD *)(v40 + 48) + 16LL * DWORD2(v40) + 8);
    v28 = *(unsigned __int16 *)(*(_QWORD *)(v40 + 48) + 16LL * DWORD2(v40));
    v46 = v26;
    if ( v26 )
    {
      v37 = v28;
      v38 = v24;
      sub_B96E90((__int64)&v46, v26, 1);
      v28 = v37;
      v24 = v38;
    }
    v29 = *(unsigned int *)(v24 + 24);
    v47 = *(_DWORD *)(v24 + 72);
    *((_QWORD *)&v35 + 1) = v9 & 0xFFFFFFFF00000000LL | v44;
    *(_QWORD *)&v35 = v45;
    v30 = sub_33FC130(v25, v29, (__int64)&v46, v28, v27, v24, v40, v35, *(_OWORD *)&v22, v42);
    v19 = v46;
    v20 = v30;
    if ( v46 )
      goto LABEL_7;
  }
  else
  {
    *(_QWORD *)&v11 = sub_383B380((__int64)a1, *(_QWORD *)v7, *(_QWORD *)(v7 + 8));
    v43 = v11;
    sub_2FE6CC0(
      (__int64)&v46,
      *a1,
      *(_QWORD *)(a1[1] + 64),
      *(unsigned __int16 *)(v10 + *(_QWORD *)(v8 + 48)),
      *(_QWORD *)(v10 + *(_QWORD *)(v8 + 48) + 8));
    v12 = a2;
    if ( (_BYTE)v46 == 1 )
    {
      v31 = sub_37AF270((__int64)a1, v8, v9, a3);
      v12 = a2;
      v45 = (unsigned __int64)v31;
      v44 = v32;
    }
    v13 = (_QWORD *)a1[1];
    v14 = *(_QWORD *)(v12 + 80);
    v15 = *(_QWORD *)(*(_QWORD *)(v43 + 48) + 16LL * DWORD2(v43) + 8);
    v16 = *(unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16LL * DWORD2(v43));
    v46 = v14;
    if ( v14 )
    {
      v39 = v16;
      v41 = v12;
      sub_B96E90((__int64)&v46, v14, 1);
      v16 = v39;
      v12 = v41;
    }
    v17 = *(_DWORD *)(v12 + 24);
    v47 = *(_DWORD *)(v12 + 72);
    *((_QWORD *)&v36 + 1) = v9 & 0xFFFFFFFF00000000LL | v44;
    *(_QWORD *)&v36 = v45;
    v18 = sub_3406EB0(v13, v17, (__int64)&v46, v16, v15, v12, v43, v36);
    v19 = v46;
    v20 = v18;
    if ( v46 )
LABEL_7:
      sub_B91220((__int64)&v46, v19);
  }
  return v20;
}
