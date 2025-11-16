// Function: sub_21E01D0
// Address: 0x21e01d0
//
__int64 __fastcall sub_21E01D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rcx
  int v10; // r8d
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  int v13; // [rsp+8h] [rbp-48h]
  _OWORD v14[4]; // [rsp+10h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD **)(a1 - 176);
  v12 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v12, v3, 2);
  v13 = *(_DWORD *)(a2 + 64);
  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 88LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (unsigned int)v7 <= 2 )
  {
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_DWORD *)(a2 + 60);
    v14[0] = _mm_loadu_si128((const __m128i *)(v5 + 120));
    v14[1] = _mm_loadu_si128((const __m128i *)v5);
    v8 = sub_1D23DE0(v4, (unsigned __int16)v7 + 4456, (__int64)&v12, v9, v10, (__int64)&v12, (__int64 *)v14, 2);
  }
  if ( v12 )
    sub_161E7C0((__int64)&v12, v12);
  return v8;
}
