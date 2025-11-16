// Function: sub_2989F30
// Address: 0x2989f30
//
__int64 __fastcall sub_2989F30(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // r15
  __int64 (__fastcall *v4)(__int64 *); // rax
  __int64 v5; // rax
  unsigned __int64 v6; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rbx
  int v9; // edx
  __int64 v10; // rcx
  __int32 v11; // eax
  __int64 v12; // r13
  unsigned int v13; // r12d
  __int32 v14; // eax
  unsigned int v15; // esi
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __int64 (__fastcall *v20)(__int64 *); // rax
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __int64 v26; // [rsp+10h] [rbp-250h]
  __int64 v27; // [rsp+18h] [rbp-248h]
  __int64 v28; // [rsp+20h] [rbp-240h]
  __int64 (__fastcall *v29)(__int64 *); // [rsp+28h] [rbp-238h]
  int v30; // [rsp+30h] [rbp-230h]
  unsigned __int64 v31; // [rsp+30h] [rbp-230h]
  __int32 v32; // [rsp+3Ch] [rbp-224h]
  unsigned int v33; // [rsp+40h] [rbp-220h]
  _QWORD *v34; // [rsp+48h] [rbp-218h]
  __int32 v35; // [rsp+48h] [rbp-218h]
  __m128i v36; // [rsp+50h] [rbp-210h] BYREF
  __m128i v37; // [rsp+60h] [rbp-200h] BYREF
  __m128i v38; // [rsp+70h] [rbp-1F0h] BYREF
  __m128i v39; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 (__fastcall *v40)(__int64 *); // [rsp+90h] [rbp-1D0h]
  __m128i v41; // [rsp+A0h] [rbp-1C0h] BYREF
  __m128i v42; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i v43; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v44; // [rsp+D0h] [rbp-190h] BYREF
  __int64 (__fastcall *v45)(__int64 *); // [rsp+E0h] [rbp-180h]

  v2 = a1;
  v3 = *(__int64 **)a2;
  v28 = *(_QWORD *)(a2 + 8);
  v4 = (__int64 (__fastcall *)(__int64 *))sub_2988260;
  if ( v28 )
    v4 = sub_2988540;
  v29 = v4;
  v5 = *v3;
  v6 = (unsigned __int64)v3 & 0xFFFFFFFFFFFFFFF9LL;
  v7 = v5 >> 2;
  v34 = (_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 48);
  v8 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
  v27 = v6 | v5 & 4;
  if ( v34 == (_QWORD *)v8 )
  {
    v32 = 0;
    v26 = 0;
    v31 = v6 | (2 * (_BYTE)v7) & 2;
    v12 = 0;
  }
  else
  {
    if ( !v8 )
LABEL_16:
      BUG();
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = v8 - 24;
    v26 = v8 - 24;
    if ( (unsigned int)(v9 - 30) > 0xA )
    {
      v26 = 0;
      v32 = 0;
    }
    else
    {
      v30 = *(unsigned __int8 *)(v8 - 24);
      v11 = sub_B46E30(v8 - 24);
      v10 = v8 - 24;
      v9 = v30;
      v32 = v11;
    }
    v31 = v6 | (2 * (_BYTE)v7) & 2;
    v12 = 0;
    if ( (unsigned int)(v9 - 30) < 0xB )
      v12 = v10;
  }
  if ( (v31 & 6) != 0 )
  {
    if ( *(_QWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 8)
                                                                     + 32LL) )
      v31 = v31 & 0xFFFFFFFFFFFFFFF9LL | 4;
    v14 = 0;
  }
  else
  {
    v13 = 0;
    do
    {
      v33 = v13;
      if ( v34 == (_QWORD *)v8 )
        goto LABEL_17;
      if ( !v8 )
        goto LABEL_16;
      if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
      {
LABEL_17:
        v14 = 0;
        if ( !v13 )
        {
LABEL_18:
          v2 = a1;
          goto LABEL_22;
        }
      }
      else
      {
        v14 = sub_B46E30(v8 - 24);
        if ( v13 == v14 )
          goto LABEL_18;
      }
      v15 = v13++;
    }
    while ( *(_QWORD *)(*(_QWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) == sub_B46EC0(v12, v15) );
    v2 = a1;
    v14 = v33;
  }
LABEL_22:
  v35 = v14;
  v41.m128i_i64[0] = v27;
  v41.m128i_i64[1] = v26;
  v42.m128i_i32[0] = v32;
  v42.m128i_i64[1] = v28;
  v43.m128i_i64[0] = v27;
  v43.m128i_i64[1] = v26;
  v44.m128i_i32[0] = v32;
  v44.m128i_i64[1] = v28;
  v45 = v29;
  sub_2988D50((__int64)&v41);
  v36.m128i_i64[1] = v12;
  v37.m128i_i32[0] = v35;
  v37.m128i_i64[1] = v28;
  v38.m128i_i64[0] = v27;
  v38.m128i_i64[1] = v26;
  v39.m128i_i32[0] = v32;
  v39.m128i_i64[1] = v28;
  v36.m128i_i64[0] = v31;
  v40 = v29;
  sub_2988D50((__int64)&v36);
  v16 = _mm_loadu_si128(&v36);
  v17 = _mm_loadu_si128(&v37);
  v18 = _mm_loadu_si128(&v38);
  *(_QWORD *)(v2 + 64) = v40;
  v19 = _mm_loadu_si128(&v39);
  v20 = v45;
  v21 = _mm_loadu_si128(&v41);
  *(__m128i *)v2 = v16;
  v22 = _mm_loadu_si128(&v42);
  v23 = _mm_loadu_si128(&v43);
  *(__m128i *)(v2 + 16) = v17;
  v24 = _mm_loadu_si128(&v44);
  *(_QWORD *)(v2 + 136) = v20;
  *(__m128i *)(v2 + 32) = v18;
  *(__m128i *)(v2 + 48) = v19;
  *(__m128i *)(v2 + 72) = v21;
  *(__m128i *)(v2 + 88) = v22;
  *(__m128i *)(v2 + 104) = v23;
  *(__m128i *)(v2 + 120) = v24;
  return v2;
}
