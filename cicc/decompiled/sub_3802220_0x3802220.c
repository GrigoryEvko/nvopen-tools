// Function: sub_3802220
// Address: 0x3802220
//
unsigned __int8 *__fastcall sub_3802220(__int64 a1, unsigned __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r15
  unsigned int v11; // ebx
  unsigned __int8 *result; // rax
  __int64 v13; // rdx
  const __m128i *v14; // rsi
  __int64 v15; // rax
  __m128i v16; // xmm1
  _QWORD *v17; // rbx
  __m128i v18; // xmm2
  __int64 v19; // rsi
  unsigned __int16 v20; // cx
  __int64 v21; // rax
  unsigned __int8 *v22; // r14
  __int64 v23; // rdx
  __m128i v24; // rcx
  __m128i v25; // kr00_16
  __int128 v26; // [rsp-10h] [rbp-D0h]
  __int64 v27; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v28; // [rsp+8h] [rbp-B8h]
  __int64 v29; // [rsp+10h] [rbp-B0h] BYREF
  int v30; // [rsp+18h] [rbp-A8h]
  __m128i v31; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+30h] [rbp-90h] BYREF
  int v33; // [rsp+38h] [rbp-88h]
  unsigned __int16 v34; // [rsp+40h] [rbp-80h] BYREF
  __int64 v35; // [rsp+48h] [rbp-78h]
  __int16 v36; // [rsp+50h] [rbp-70h]
  __int64 v37; // [rsp+58h] [rbp-68h]
  _OWORD v38[6]; // [rsp+60h] [rbp-60h] BYREF

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 40);
  if ( v3 > 239 )
  {
    if ( (unsigned int)(v3 - 242) > 1 )
    {
LABEL_4:
      v30 = 0;
      HIWORD(v11) = 0;
      v31.m128i_i32[2] = 0;
      v29 = 0;
      v31.m128i_i64[0] = 0;
      sub_375E6F0(a1, *(_QWORD *)v4, *(_QWORD *)(v4 + 8), (__int64)&v29, (__int64)&v31);
      v6 = *(_QWORD *)(a2 + 48);
      v7 = *(_QWORD *)(a2 + 80);
      v8 = *(_QWORD **)(a1 + 8);
      v9 = *(_QWORD *)(a2 + 40);
      v10 = *(_QWORD *)(v6 + 8);
      LOWORD(v11) = *(_WORD *)v6;
      *(_QWORD *)&v38[0] = v7;
      if ( v7 )
      {
        v27 = v9;
        sub_B96E90((__int64)v38, v7, 1);
        v9 = v27;
      }
      DWORD2(v38[0]) = *(_DWORD *)(a2 + 72);
      result = sub_3406EB0(v8, 0xE6u, (__int64)v38, v11, v10, v5, *(_OWORD *)&v31, *(_OWORD *)(v9 + 40));
      if ( *(_QWORD *)&v38[0] )
      {
        v28 = result;
        sub_B91220((__int64)v38, *(__int64 *)&v38[0]);
        return v28;
      }
      return result;
    }
  }
  else if ( v3 <= 237 && (unsigned int)(v3 - 101) > 0x2F )
  {
    goto LABEL_4;
  }
  v30 = 0;
  v31.m128i_i32[2] = 0;
  v29 = 0;
  v31.m128i_i64[0] = 0;
  sub_375E6F0(a1, *(_QWORD *)(v4 + 40), *(_QWORD *)(v4 + 48), (__int64)&v29, (__int64)&v31);
  v13 = *(_QWORD *)(a2 + 48);
  v14 = *(const __m128i **)(a2 + 40);
  v15 = *(_QWORD *)(v31.m128i_i64[0] + 48) + 16LL * v31.m128i_u32[2];
  if ( *(_WORD *)v13 == *(_WORD *)v15 && (*(_QWORD *)(v15 + 8) == *(_QWORD *)(v13 + 8) || *(_WORD *)v15) )
  {
    sub_3760E70(a1, a2, 1, v14->m128i_i64[0], v14->m128i_i64[1]);
    v25 = v31;
  }
  else
  {
    v16 = _mm_loadu_si128(&v31);
    v17 = *(_QWORD **)(a1 + 8);
    v38[0] = _mm_loadu_si128(v14);
    v38[1] = v16;
    v18 = _mm_loadu_si128(v14 + 5);
    v19 = *(_QWORD *)(a2 + 80);
    v38[2] = v18;
    v20 = *(_WORD *)v13;
    v21 = *(_QWORD *)(v13 + 8);
    v32 = v19;
    v34 = v20;
    v36 = 1;
    v35 = v21;
    v37 = 0;
    if ( v19 )
      sub_B96E90((__int64)&v32, v19, 1);
    *((_QWORD *)&v26 + 1) = 3;
    *(_QWORD *)&v26 = v38;
    v33 = *(_DWORD *)(a2 + 72);
    v22 = sub_3411BE0(v17, 0x91u, (__int64)&v32, &v34, 2, (__int64)&v32, v26);
    v24.m128i_i64[1] = v23;
    if ( v32 )
      sub_B91220((__int64)&v32, v32);
    sub_3760E70(a1, a2, 1, (unsigned __int64)v22, 1);
    v24.m128i_i64[0] = (__int64)v22;
    v25 = v24;
  }
  sub_3760E70(a1, a2, 0, v25.m128i_u64[0], v25.m128i_i64[1]);
  return 0;
}
