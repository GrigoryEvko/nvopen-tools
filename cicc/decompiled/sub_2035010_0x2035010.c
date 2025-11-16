// Function: sub_2035010
// Address: 0x2035010
//
__int64 __fastcall sub_2035010(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rsi
  __int64 *v12; // r13
  __int32 v13; // edx
  const void **v14; // r15
  unsigned int v15; // ebx
  __int64 v16; // r12
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  int v19; // [rsp+18h] [rbp-68h]
  __m128i v20[6]; // [rsp+20h] [rbp-60h]

  v7 = a3;
  v8 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  v9 = sub_2032580(a1, *(_QWORD *)v8, *(_QWORD *)(v8 + 8));
  v10 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v11 = *(_QWORD *)(a2 + 72);
  v20[0] = v10;
  v12 = *(__int64 **)(a1 + 8);
  v20[v7].m128i_i64[0] = v9;
  v20[v7].m128i_i32[2] = v13;
  v14 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v15 = **(unsigned __int8 **)(a2 + 40);
  v18 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v18, v11, 2);
  v19 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D309E0(v12, 134, (__int64)&v18, v15, v14, 0, *(double *)v10.m128i_i64, a5, a6, *(_OWORD *)v20);
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v16;
}
