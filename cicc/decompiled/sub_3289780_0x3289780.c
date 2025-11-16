// Function: sub_3289780
// Address: 0x3289780
//
__int64 __fastcall sub_3289780(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        char a6,
        __int128 a7,
        char a8)
{
  int v9; // r14d
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // rax
  unsigned __int16 v19; // r15
  __int64 v20; // r10
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rsi
  __int128 v27; // rax
  int v28; // r9d
  bool v29; // al
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned __int16 v33; // ax
  __int128 v34; // [rsp-20h] [rbp-90h]
  __int128 v35; // [rsp-10h] [rbp-80h]
  int v36; // [rsp+0h] [rbp-70h]
  int v37; // [rsp+0h] [rbp-70h]
  int v38; // [rsp+8h] [rbp-68h]
  __m128i v39; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int16 v40; // [rsp+20h] [rbp-50h] BYREF
  __int64 v41; // [rsp+28h] [rbp-48h]

  v9 = a4;
  if ( a8 )
  {
    v39 = _mm_loadu_si128((const __m128i *)&a7);
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
    v15 = *(_WORD *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    v39.m128i_i16[0] = v15;
    v39.m128i_i64[1] = v16;
  }
  v12 = sub_3288BD0(*a1, a4, v39.m128i_u32[0], v39.m128i_u64[1], a2, a3, 0, a5);
  if ( v12 )
    return v12;
  if ( a6 )
    return v12;
  v36 = v11;
  v17 = sub_33E0A10(*a1, a2, a3, 0, 0, v11);
  v12 = 0;
  if ( !v17 )
    return v12;
  *((_QWORD *)&v35 + 1) = a3;
  *(_QWORD *)&v35 = a2;
  v18 = sub_33FAF80(*a1, 199, v9, v39.m128i_i32[0], v39.m128i_i32[2], v36, v35);
  v19 = v39.m128i_i16[0];
  v20 = *a1;
  v21 = v18;
  v23 = v22;
  if ( v39.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v39.m128i_i16[0] - 17) <= 0xD3u )
    {
      v19 = word_4456580[v39.m128i_u16[0] - 1];
      v24 = 0;
      goto LABEL_11;
    }
  }
  else
  {
    v37 = *a1;
    v29 = sub_30070B0((__int64)&v39);
    LODWORD(v20) = v37;
    if ( v29 )
    {
      v33 = sub_3009970((__int64)&v39, 199, v30, v31, v32);
      LODWORD(v20) = v37;
      v19 = v33;
      goto LABEL_11;
    }
  }
  v24 = v39.m128i_i64[1];
LABEL_11:
  v40 = v19;
  v41 = v24;
  if ( v19 )
  {
    if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
      BUG();
    v26 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
  }
  else
  {
    v38 = v20;
    v25 = sub_3007260((__int64)&v40);
    LODWORD(v20) = v38;
    LODWORD(v26) = v25;
  }
  *(_QWORD *)&v27 = sub_3400BD0(v20, (int)v26 - 1, v9, v39.m128i_i32[0], v39.m128i_i32[2], 0, 0);
  *((_QWORD *)&v34 + 1) = v23;
  *(_QWORD *)&v34 = v21;
  return sub_3406EB0(*a1, 57, v9, v39.m128i_i32[0], v39.m128i_i32[2], v28, v27, v34);
}
