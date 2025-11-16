// Function: sub_2E7AC20
// Address: 0x2e7ac20
//
__int64 __fastcall sub_2E7AC20(__int64 a1, __int64 a2, const __m128i *a3, int a4)
{
  unsigned __int8 v4; // bl
  __int64 v5; // rax
  unsigned __int8 v6; // cl
  int v7; // r15d
  int v8; // r13d
  unsigned __int8 v9; // r14
  unsigned __int8 v10; // bl
  int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // r12
  unsigned __int8 v16; // [rsp+Fh] [rbp-71h]
  __m128i v17; // [rsp+10h] [rbp-70h]
  int v18; // [rsp+20h] [rbp-60h]
  _QWORD v19[4]; // [rsp+30h] [rbp-50h] BYREF

  v4 = *(_BYTE *)(a2 + 37);
  v5 = a3[1].m128i_i64[0];
  v6 = *(_BYTE *)(a2 + 34);
  memset(v19, 0, sizeof(v19));
  v7 = *(unsigned __int16 *)(a2 + 32);
  v8 = *(unsigned __int8 *)(a2 + 36);
  v9 = v4;
  v18 = v5;
  v10 = v4 >> 4;
  v16 = v6;
  v11 = v9 & 0xF;
  v17 = _mm_loadu_si128(a3);
  v12 = sub_A777F0(0x58u, (__int64 *)(a1 + 128));
  v13 = v12;
  if ( v12 )
    sub_2EAC440(v12, v7, a4, v16, (unsigned int)v19, 0, v17.m128i_i32[0], v17.m128i_i32[2], v18, v8, v11, v10);
  return v13;
}
