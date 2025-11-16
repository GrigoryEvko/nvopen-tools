// Function: sub_21E0290
// Address: 0x21e0290
//
__int64 __fastcall sub_21E0290(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // r13
  const __m128i *v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // r12
  int v9; // r8d
  __m128i v10; // xmm1
  int v11; // esi
  __int64 v12; // rcx
  __int64 v14; // [rsp+0h] [rbp-60h] BYREF
  int v15; // [rsp+8h] [rbp-58h]
  _OWORD v16[5]; // [rsp+10h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD **)(a1 - 176);
  v14 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v14, v3, 2);
  v5 = *(const __m128i **)(a2 + 32);
  v15 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(v5[7].m128i_i64[1] + 88);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (unsigned int)v7 <= 3 )
  {
    v9 = *(_DWORD *)(a2 + 60);
    v10 = _mm_loadu_si128(v5 + 10);
    v11 = dword_435EE30[(unsigned int)v7];
    v16[0] = _mm_loadu_si128(v5 + 5);
    v12 = *(_QWORD *)(a2 + 40);
    v16[1] = v10;
    v16[2] = _mm_loadu_si128(v5);
    v8 = sub_1D23DE0(v4, v11, (__int64)&v14, v12, v9, (__int64)&v14, (__int64 *)v16, 3);
  }
  if ( v14 )
    sub_161E7C0((__int64)&v14, v14);
  return v8;
}
