// Function: sub_3794BC0
// Address: 0x3794bc0
//
unsigned __int8 *__fastcall sub_3794BC0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // r12
  __m128i v9; // xmm0
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int16 *v13; // rbx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r8
  _QWORD *v17; // r12
  __int128 v18; // rax
  __int64 v19; // r9
  unsigned __int8 *v20; // rax
  unsigned int v21; // edx
  __int64 v22; // r12
  __int128 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r9
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // edx
  unsigned __int16 *v29; // rdx
  _QWORD *v30; // r14
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 v35; // rsi
  unsigned int v36; // esi
  unsigned __int8 *v37; // r12
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned int v41; // edx
  unsigned int v42; // edx
  __int64 v43; // rdx
  __int64 v44; // [rsp+8h] [rbp-D8h]
  _QWORD *v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+20h] [rbp-C0h]
  unsigned int v48; // [rsp+20h] [rbp-C0h]
  unsigned int v49; // [rsp+28h] [rbp-B8h]
  __int16 v50; // [rsp+2Ah] [rbp-B6h]
  __m128i v51; // [rsp+30h] [rbp-B0h]
  __int64 v53; // [rsp+70h] [rbp-70h] BYREF
  int v54; // [rsp+78h] [rbp-68h]
  __int64 v55; // [rsp+80h] [rbp-60h] BYREF
  int v56; // [rsp+88h] [rbp-58h]
  __int16 v57; // [rsp+90h] [rbp-50h] BYREF
  __int64 v58; // [rsp+98h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 80);
  v53 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v53, v4, 1);
  v5 = a1[1];
  v6 = *a1;
  v54 = *(_DWORD *)(a2 + 72);
  v7 = *(__int64 **)(a2 + 40);
  v8 = *v7;
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = 16LL * *((unsigned int *)v7 + 2);
  v51 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  sub_2FE6CC0(
    (__int64)&v57,
    v6,
    *(_QWORD *)(v5 + 64),
    *(unsigned __int16 *)(v10 + *(_QWORD *)(*v7 + 48)),
    *(_QWORD *)(v10 + *(_QWORD *)(*v7 + 48) + 8));
  if ( (_BYTE)v57 == 5 )
  {
    v40 = sub_37946F0((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    v48 = v41;
    v22 = v40;
    v24 = v51.m128i_i64[0];
    v51.m128i_i64[0] = sub_37946F0((__int64)a1, v51.m128i_u64[0], v51.m128i_i64[1]);
    v51.m128i_i64[1] = v42 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v13 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + v10);
    v14 = *v13;
    v15 = *((_QWORD *)v13 + 1);
    v57 = v14;
    v58 = v15;
    if ( (_WORD)v14 )
    {
      v16 = 0;
      LOWORD(v14) = word_4456580[v14 - 1];
    }
    else
    {
      v14 = sub_3009970((__int64)&v57, v6, v15, v11, v12);
      HIWORD(v2) = HIWORD(v14);
      v16 = v39;
    }
    v17 = (_QWORD *)a1[1];
    LOWORD(v2) = v14;
    v47 = v16;
    *(_QWORD *)&v18 = sub_3400EE0((__int64)v17, 0, (__int64)&v53, 0, v9);
    v44 = v47;
    v20 = sub_3406EB0(v17, 0x9Eu, (__int64)&v53, v2, v47, v19, *(_OWORD *)&v9, v18);
    v48 = v21;
    v22 = (__int64)v20;
    v45 = (_QWORD *)a1[1];
    *(_QWORD *)&v23 = sub_3400EE0((__int64)v45, 0, (__int64)&v53, 0, v9);
    v24 = 158;
    v51.m128i_i64[0] = (__int64)sub_3406EB0(v45, 0x9Eu, (__int64)&v53, v2, v44, v25, *(_OWORD *)&v51, v23);
    v51.m128i_i64[1] = v28 | v51.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  v29 = *(unsigned __int16 **)(a2 + 48);
  v30 = (_QWORD *)a1[1];
  v31 = *v29;
  v32 = *((_QWORD *)v29 + 1);
  v57 = v31;
  v58 = v32;
  if ( (_WORD)v31 )
  {
    v33 = 0;
    LOWORD(v31) = word_4456580[v31 - 1];
  }
  else
  {
    v31 = sub_3009970((__int64)&v57, v24, v32, 0xFFFFFFFF00000000LL, v26);
    v50 = HIWORD(v31);
    v33 = v43;
  }
  HIWORD(v34) = v50;
  v35 = *(_QWORD *)(a2 + 80);
  LOWORD(v34) = v31;
  v55 = v35;
  v49 = v34;
  if ( v35 )
  {
    v46 = v33;
    sub_B96E90((__int64)&v55, v35, 1);
    v33 = v46;
  }
  v36 = *(_DWORD *)(a2 + 24);
  v56 = *(_DWORD *)(a2 + 72);
  v37 = sub_3406EB0(
          v30,
          v36,
          (__int64)&v55,
          v49,
          v33,
          v27,
          __PAIR128__(v48 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL, v22),
          *(_OWORD *)&v51);
  if ( v55 )
    sub_B91220((__int64)&v55, v55);
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v37;
}
