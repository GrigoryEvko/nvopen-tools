// Function: sub_21E3260
// Address: 0x21e3260
//
__int64 __fastcall sub_21E3260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // r14
  __m128i v10; // xmm1
  __int64 v11; // rax
  bool v12; // cf
  __int16 v13; // r12
  __int64 v14; // rcx
  int v15; // r8d
  __int64 v16; // r12
  __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  int v19; // [rsp+8h] [rbp-48h]
  _OWORD v20[4]; // [rsp+10h] [rbp-40h] BYREF

  v7 = *(const __m128i **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_QWORD **)(a1 - 176);
  v20[0] = _mm_loadu_si128(v7 + 5);
  v10 = _mm_loadu_si128(v7);
  v11 = *(_QWORD *)(a1 + 16);
  v18 = v8;
  v12 = *(_BYTE *)(v11 + 936) == 0;
  v20[1] = v10;
  v13 = 3143 - (v12 - 1);
  if ( v8 )
    sub_1623A60((__int64)&v18, v8, 2);
  v14 = *(_QWORD *)(a2 + 40);
  v15 = *(_DWORD *)(a2 + 60);
  v19 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D23DE0(v9, v13, (__int64)&v18, v14, v15, a6, (__int64 *)v20, 2);
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v16;
}
