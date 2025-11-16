// Function: sub_21E0020
// Address: 0x21e0020
//
__int64 __fastcall sub_21E0020(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // r12
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  int v12; // esi
  __int64 v13; // rcx
  int v14; // r8d
  __int64 v16; // [rsp+0h] [rbp-80h] BYREF
  int v17; // [rsp+8h] [rbp-78h]
  _OWORD v18[7]; // [rsp+10h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD **)(a1 - 176);
  v16 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v16, v3, 2);
  v5 = *(_QWORD *)(a2 + 32);
  v17 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 120) + 88LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (unsigned int)v7 <= 3 )
  {
    v9 = _mm_loadu_si128((const __m128i *)(v5 + 160));
    v10 = _mm_loadu_si128((const __m128i *)(v5 + 200));
    v11 = _mm_loadu_si128((const __m128i *)(v5 + 240));
    v12 = dword_435EE40[(unsigned int)v7];
    v18[0] = _mm_loadu_si128((const __m128i *)(v5 + 80));
    v18[1] = v9;
    v13 = *(_QWORD *)(a2 + 40);
    v14 = *(_DWORD *)(a2 + 60);
    v18[2] = v10;
    v18[3] = v11;
    v18[4] = _mm_loadu_si128((const __m128i *)v5);
    v8 = sub_1D23DE0(v4, v12, (__int64)&v16, v13, v14, (__int64)&v16, (__int64 *)v18, 5);
  }
  if ( v16 )
    sub_161E7C0((__int64)&v16, v16);
  return v8;
}
