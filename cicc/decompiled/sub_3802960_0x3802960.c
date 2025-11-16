// Function: sub_3802960
// Address: 0x3802960
//
__int64 __fastcall sub_3802960(_QWORD *a1, unsigned __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rsi
  char v7; // bl
  unsigned int v8; // edx
  unsigned int v10; // [rsp+Ch] [rbp-64h] BYREF
  __m128i v11; // [rsp+10h] [rbp-60h] BYREF
  __m128i v12; // [rsp+20h] [rbp-50h] BYREF
  __m128i v13; // [rsp+30h] [rbp-40h] BYREF
  __int64 v14; // [rsp+40h] [rbp-30h] BYREF
  int v15; // [rsp+48h] [rbp-28h]

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 40);
  if ( v3 > 239 )
  {
    if ( (unsigned int)(v3 - 242) > 1 )
    {
LABEL_4:
      v13.m128i_i32[2] = 0;
      v5 = 80;
      v13.m128i_i64[0] = 0;
      v11 = _mm_loadu_si128((const __m128i *)v4);
      v12 = _mm_loadu_si128((const __m128i *)(v4 + 40));
      goto LABEL_5;
    }
  }
  else if ( v3 <= 237 && (unsigned int)(v3 - 101) > 0x2F )
  {
    goto LABEL_4;
  }
  v5 = 120;
  v11 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v12 = _mm_loadu_si128((const __m128i *)(v4 + 80));
  v13 = _mm_loadu_si128((const __m128i *)v4);
LABEL_5:
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v3 == 148;
  v8 = *(_DWORD *)(*(_QWORD *)(v4 + v5) + 96LL);
  v14 = v6;
  v10 = v8;
  if ( v6 )
    sub_B96E90((__int64)&v14, v6, 1);
  v15 = *(_DWORD *)(a2 + 72);
  sub_38014E0(a1, (unsigned __int64 *)&v11, (__int64)&v12, &v10, (__int64)&v14, (__int64)&v13, v7);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
  if ( !v13.m128i_i64[0] )
    return v11.m128i_i64[0];
  sub_3760E70((__int64)a1, a2, 0, v11.m128i_u64[0], v11.m128i_i64[1]);
  sub_3760E70((__int64)a1, a2, 1, v13.m128i_u64[0], v13.m128i_i64[1]);
  return 0;
}
