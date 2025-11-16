// Function: sub_3797EA0
// Address: 0x3797ea0
//
unsigned __int8 *__fastcall sub_3797EA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // ebx
  __int64 v7; // rax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r8
  int v22; // eax
  unsigned int v23; // ecx
  __int128 v24; // rax
  __int64 v25; // r9
  unsigned __int8 *v26; // rax
  unsigned int v27; // edx
  __int128 v28; // rax
  __int64 v29; // r9
  __int64 v30; // r9
  unsigned int v31; // edx
  int v32; // r9d
  _DWORD *v33; // r14
  unsigned __int16 v34; // si
  unsigned int v35; // eax
  unsigned __int8 *v36; // r14
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned int v40; // edx
  unsigned int v41; // edx
  __int64 v42; // rdx
  __int64 v43; // [rsp+0h] [rbp-E0h]
  _QWORD *v44; // [rsp+8h] [rbp-D8h]
  __int64 v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  _QWORD *v48; // [rsp+20h] [rbp-C0h]
  unsigned int v49; // [rsp+20h] [rbp-C0h]
  unsigned int v50; // [rsp+28h] [rbp-B8h]
  __int16 v51; // [rsp+2Ah] [rbp-B6h]
  __int128 v52; // [rsp+30h] [rbp-B0h]
  char v53; // [rsp+30h] [rbp-B0h]
  __m128i v54; // [rsp+70h] [rbp-70h] BYREF
  __int64 v55; // [rsp+80h] [rbp-60h] BYREF
  int v56; // [rsp+88h] [rbp-58h]
  __m128i v57; // [rsp+90h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a2 + 40);
  v8 = _mm_loadu_si128((const __m128i *)v7);
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * *(unsigned int *)(v7 + 8);
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v54.m128i_i16[0] = v11;
  v13 = *(unsigned __int16 **)(a2 + 48);
  v54.m128i_i64[1] = v12;
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v57.m128i_i16[0] = v14;
  v57.m128i_i64[1] = v15;
  if ( (_WORD)v14 )
  {
    v45 = 0;
    LOWORD(v14) = word_4456580[v14 - 1];
  }
  else
  {
    v14 = sub_3009970((__int64)&v57, a2, v15, a4, a5);
    v45 = v42;
    HIWORD(v5) = HIWORD(v14);
  }
  v16 = *(_QWORD *)(a2 + 80);
  LOWORD(v5) = v14;
  v55 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v55, v16, 1);
  v17 = *a1;
  v56 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v57, v17, *(_QWORD *)(a1[1] + 64), v54.m128i_u16[0], v54.m128i_i64[1]);
  if ( v57.m128i_i8[0] == 5 )
  {
    v39 = sub_37946F0((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1]);
    v49 = v40;
    v47 = v39;
    *(_QWORD *)&v52 = sub_37946F0((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    *((_QWORD *)&v52 + 1) = v41 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else
  {
    if ( v54.m128i_i16[0] )
    {
      v21 = 0;
      LOWORD(v22) = word_4456580[v54.m128i_u16[0] - 1];
    }
    else
    {
      v22 = sub_3009970((__int64)&v54, v17, v18, v19, v20);
      v51 = HIWORD(v22);
      v21 = v38;
    }
    HIWORD(v23) = v51;
    v46 = v21;
    LOWORD(v23) = v22;
    v48 = (_QWORD *)a1[1];
    v50 = v23;
    *(_QWORD *)&v24 = sub_3400EE0((__int64)v48, 0, (__int64)&v55, 0, v8);
    v43 = v46;
    v26 = sub_3406EB0(v48, 0x9Eu, (__int64)&v55, v50, v46, v25, *(_OWORD *)&v8, v24);
    v49 = v27;
    v44 = (_QWORD *)a1[1];
    v47 = (__int64)v26;
    *(_QWORD *)&v28 = sub_3400EE0((__int64)v44, 0, (__int64)&v55, 0, v8);
    *(_QWORD *)&v52 = sub_3406EB0(v44, 0x9Eu, (__int64)&v55, v50, v43, v29, *(_OWORD *)&v9, v28);
    *((_QWORD *)&v52 + 1) = v31 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  sub_340F900(
    (_QWORD *)a1[1],
    0xD0u,
    (__int64)&v55,
    2u,
    0,
    v30,
    __PAIR128__(v49 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL, v47),
    v52,
    *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v33 = (_DWORD *)*a1;
  v57 = _mm_loadu_si128(&v54);
  if ( v54.m128i_i16[0] )
  {
    v34 = v54.m128i_i16[0] - 17;
    if ( (unsigned __int16)(v54.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v54.m128i_i16[0] - 126) > 0x31u )
    {
      if ( v34 > 0xD3u )
      {
LABEL_13:
        v35 = v33[15];
        goto LABEL_14;
      }
      goto LABEL_20;
    }
    if ( v34 <= 0xD3u )
    {
LABEL_20:
      v35 = v33[17];
      goto LABEL_14;
    }
  }
  else
  {
    v53 = sub_3007030((__int64)&v57);
    if ( sub_30070B0((__int64)&v57) )
      goto LABEL_20;
    if ( !v53 )
      goto LABEL_13;
  }
  v35 = v33[16];
LABEL_14:
  if ( v35 > 2 )
    BUG();
  v36 = sub_33FAF80(a1[1], 215 - v35, (__int64)&v55, v5, v45, v32, v8);
  if ( v55 )
    sub_B91220((__int64)&v55, v55);
  return v36;
}
