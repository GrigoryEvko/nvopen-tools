// Function: sub_2135230
// Address: 0x2135230
//
__int64 __fastcall sub_2135230(__int64 a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  __m128i v5; // xmm0
  unsigned __int8 *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rax
  int v9; // eax
  __m128i *v10; // r10
  __int64 v11; // r9
  int v12; // r13d
  __int64 v13; // r11
  __int64 v14; // r12
  __int64 v16; // [rsp+0h] [rbp-80h]
  __int64 v17; // [rsp+8h] [rbp-78h]
  __m128i *v18; // [rsp+8h] [rbp-78h]
  __m128i v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 v20; // [rsp+20h] [rbp-60h] BYREF
  int v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF

  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v6 = *(unsigned __int8 **)(a2 + 40);
  v19 = v5;
  v7 = *v6;
  v17 = *((_QWORD *)v6 + 1);
  v8 = *(_QWORD *)(v5.m128i_i64[0] + 40) + 16LL * v5.m128i_u32[2];
  v9 = sub_1F40200(*(_BYTE *)v8, *(_QWORD *)(v8 + 8), v7);
  v10 = *(__m128i **)a1;
  v11 = v17;
  v12 = v9;
  v20 = *(_QWORD *)(a2 + 72);
  if ( v20 )
  {
    v16 = v17;
    v18 = v10;
    sub_1623A60((__int64)&v20, v20, 2);
    v11 = v16;
    v10 = v18;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v21 = *(_DWORD *)(a2 + 64);
  sub_20BE530((__int64)&v22, v10, v13, v12, v7, v11, v5, a4, a5, (__int64)&v19, 1u, 1u, (__int64)&v20, 0, 1);
  v14 = v22;
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v14;
}
