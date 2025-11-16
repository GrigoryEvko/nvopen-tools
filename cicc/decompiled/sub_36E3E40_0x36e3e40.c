// Function: sub_36E3E40
// Address: 0x36e3e40
//
__int64 __fastcall sub_36E3E40(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rsi
  unsigned int v8; // r13d
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  int v19; // [rsp+8h] [rbp-48h]
  _OWORD v20[4]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v18 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v18, v4, 1);
  v5 = *(_QWORD *)(a2 + 40);
  v19 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 96LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (unsigned int)v7 <= 2 )
  {
    v10 = *(_QWORD *)(a2 + 48);
    v11 = *(unsigned int *)(a2 + 68);
    v12 = *(_QWORD **)(a1 + 64);
    v20[0] = _mm_loadu_si128((const __m128i *)(v5 + 120));
    v20[1] = _mm_loadu_si128((const __m128i *)v5);
    v13 = sub_33E66D0(v12, (int)v7 + 5595, (__int64)&v18, v10, v11, (__int64)&v18, (unsigned __int64 *)v20, 2);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v13, v14, v15, v16);
    v17 = v13;
    v8 = 1;
    sub_3421DB0(v17);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v8;
}
