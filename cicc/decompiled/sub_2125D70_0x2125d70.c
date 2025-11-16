// Function: sub_2125D70
// Address: 0x2125d70
//
__int64 __fastcall sub_2125D70(__int64 a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  unsigned int v6; // r14d
  __m128i v7; // xmm0
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  char v11; // cl
  __int64 v12; // r9
  char v13; // si
  int v14; // eax
  __int64 v15; // rsi
  __m128i *v16; // r11
  int v17; // ecx
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 v21; // rsi
  __int64 *v22; // r15
  __int32 v23; // edx
  int v24; // r8d
  int v25; // r9d
  unsigned __int32 v26; // edx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  int v32; // [rsp+Ch] [rbp-A4h]
  __m128i *v33; // [rsp+10h] [rbp-A0h]
  __int64 v34; // [rsp+18h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-90h]
  __m128i v36; // [rsp+40h] [rbp-70h] BYREF
  __int64 v37; // [rsp+50h] [rbp-60h] BYREF
  int v38; // [rsp+58h] [rbp-58h]
  __int64 v39; // [rsp+60h] [rbp-50h] BYREF
  __int64 v40; // [rsp+68h] [rbp-48h]
  __int64 v41; // [rsp+70h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v39,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v40;
  v34 = v41;
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v36 = v7;
  v8 = *(_QWORD *)(v7.m128i_i64[0] + 40) + 16LL * v7.m128i_u32[2];
  v9 = *(_BYTE *)v8;
  if ( *(_BYTE *)v8 == 8 && **(_BYTE **)(a2 + 40) != 9 )
  {
    v21 = *(_QWORD *)(a2 + 72);
    v22 = *(__int64 **)(a1 + 8);
    v39 = v21;
    if ( v21 )
      sub_1623A60((__int64)&v39, v21, 2);
    LODWORD(v40) = *(_DWORD *)(a2 + 64);
    v36.m128i_i64[0] = sub_1D309E0(
                         v22,
                         157,
                         (__int64)&v39,
                         9,
                         0,
                         0,
                         *(double *)v7.m128i_i64,
                         *(double *)a4.m128i_i64,
                         *(double *)a5.m128i_i64,
                         *(_OWORD *)&v36);
    v36.m128i_i32[2] = v23;
    if ( v39 )
      sub_161E7C0((__int64)&v39, v39);
    sub_1F40D10((__int64)&v39, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), 9, 0);
    if ( (_BYTE)v39 == 3 )
    {
      v30 = v36.m128i_i64[0];
      *(_DWORD *)(v36.m128i_i64[0] + 28) = 0;
      v31 = *(unsigned int *)(a1 + 1376);
      if ( (unsigned int)v31 >= *(_DWORD *)(a1 + 1380) )
      {
        sub_16CD150(a1 + 1368, (const void *)(a1 + 1384), 0, 8, v24, v25);
        v31 = *(unsigned int *)(a1 + 1376);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1368) + 8 * v31) = v30;
      ++*(_DWORD *)(a1 + 1376);
    }
    v8 = *(_QWORD *)(v36.m128i_i64[0] + 40) + 16LL * v36.m128i_u32[2];
    v9 = *(_BYTE *)v8;
  }
  sub_1F40D10((__int64)&v39, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v9, *(_QWORD *)(v8 + 8));
  if ( (_BYTE)v39 == 8 )
  {
    v35 = sub_2125740(a1, v36.m128i_u64[0], v36.m128i_i64[1]);
    v36.m128i_i32[2] = v26;
    v27 = *(_QWORD *)(a2 + 40);
    v36.m128i_i64[0] = v35;
    v13 = *(_BYTE *)v27;
    v28 = *(_QWORD *)(v27 + 8);
    v29 = *(_QWORD *)(v35 + 40) + 16LL * v26;
    v11 = *(_BYTE *)v29;
    v12 = *(_QWORD *)(v29 + 8);
    if ( *(_BYTE *)v29 == v13 )
    {
      if ( v11 || v12 == v28 )
        return sub_200D2A0(
                 a1,
                 v36.m128i_i64[0],
                 v36.m128i_i64[1],
                 *(double *)v7.m128i_i64,
                 *(double *)a4.m128i_i64,
                 *(double *)a5.m128i_i64);
      v11 = 0;
    }
  }
  else
  {
    v10 = *(_QWORD *)(v36.m128i_i64[0] + 40) + 16LL * v36.m128i_u32[2];
    v11 = *(_BYTE *)v10;
    v12 = *(_QWORD *)(v10 + 8);
    v13 = **(_BYTE **)(a2 + 40);
  }
  v14 = sub_1F3FE80(v11, v12, v13);
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *(__m128i **)a1;
  v17 = v14;
  v37 = v15;
  if ( v15 )
  {
    v33 = v16;
    v32 = v14;
    sub_1623A60((__int64)&v37, v15, 2);
    v17 = v32;
    v16 = v33;
  }
  v18 = *(_QWORD *)(a1 + 8);
  v38 = *(_DWORD *)(a2 + 64);
  sub_20BE530((__int64)&v39, v16, v18, v17, v6, v34, v7, a4, a5, (__int64)&v36, 1u, 0, (__int64)&v37, 0, 1);
  v19 = v39;
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
  return v19;
}
