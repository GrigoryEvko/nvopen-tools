// Function: sub_21DF500
// Address: 0x21df500
//
__int64 __fastcall sub_21DF500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rax
  __m128i v11; // xmm1
  bool v12; // zf
  int v13; // r8d
  __int64 v14; // r12
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  int v17; // [rsp+8h] [rbp-58h]
  _OWORD v18[5]; // [rsp+10h] [rbp-50h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("match instruction not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD **)(a1 - 176);
  v16 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v16, v7, 2);
  v9 = *(_QWORD *)(a2 + 40);
  v17 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(a2 + 32);
  v11 = _mm_loadu_si128((const __m128i *)(v10 + 120));
  v12 = **(_BYTE **)(*(_QWORD *)(v10 + 120) + 40LL) == 5;
  v18[0] = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v18[1] = v11;
  v13 = *(_DWORD *)(a2 + 60);
  v18[2] = _mm_loadu_si128((const __m128i *)v10);
  v14 = sub_1D23DE0(v8, 2 * !v12 + 3138, (__int64)&v16, v9, v13, a6, (__int64 *)v18, 3);
  if ( v16 )
    sub_161E7C0((__int64)&v16, v16);
  return v14;
}
