// Function: sub_211EF90
// Address: 0x211ef90
//
__int64 __fastcall sub_211EF90(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int8 *v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // r14d
  __int64 v9; // r15
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // r14
  __int64 v14; // [rsp+0h] [rbp-60h] BYREF
  int v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF

  v6 = *(unsigned __int8 **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v14 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v14, v7, 2);
  v15 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v11 = sub_1F40000(*(_BYTE *)v10, *(_QWORD *)(v10 + 8), v8);
  sub_20BE530(
    (__int64)&v16,
    *(__m128i **)a1,
    *(_QWORD *)(a1 + 8),
    v11,
    v8,
    v9,
    a3,
    a4,
    a5,
    *(_QWORD *)(a2 + 32),
    1u,
    0,
    (__int64)&v14,
    0,
    1);
  v12 = v16;
  if ( v14 )
    sub_161E7C0((__int64)&v14, v14);
  return v12;
}
