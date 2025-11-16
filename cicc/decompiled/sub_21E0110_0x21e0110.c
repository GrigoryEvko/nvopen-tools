// Function: sub_21E0110
// Address: 0x21e0110
//
__int64 __fastcall sub_21E0110(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // r13
  __int64 v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // r12
  int v9; // r8d
  __m128i v10; // xmm1
  __int64 v11; // rcx
  __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  int v14; // [rsp+8h] [rbp-48h]
  _OWORD v15[4]; // [rsp+10h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD **)(a1 - 176);
  v13 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v13, v3, 2);
  v5 = *(_QWORD *)(a2 + 32);
  v14 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 88LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (_DWORD)v7 == 3 )
  {
    v9 = *(_DWORD *)(a2 + 60);
    v15[0] = _mm_loadu_si128((const __m128i *)(v5 + 120));
    v10 = _mm_loadu_si128((const __m128i *)v5);
    v11 = *(_QWORD *)(a2 + 40);
    v15[1] = v10;
    v8 = sub_1D23DE0(v4, 4455, (__int64)&v13, v11, v9, (__int64)&v13, (__int64 *)v15, 2);
  }
  if ( v13 )
    sub_161E7C0((__int64)&v13, v13);
  return v8;
}
