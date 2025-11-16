// Function: sub_211F070
// Address: 0x211f070
//
__int64 __fastcall sub_211F070(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  char *v6; // rax
  __int64 v7; // rsi
  char v8; // r14
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // r14
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  int v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h] BYREF

  v6 = *(char **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *v6;
  v13 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v13, v7, 2);
  v14 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v10 = sub_1F40100(*(_BYTE *)v9, *(_QWORD *)(v9 + 8), v8);
  sub_20BE530(
    (__int64)&v15,
    *(__m128i **)a1,
    *(_QWORD *)(a1 + 8),
    v10,
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
    a3,
    a4,
    a5,
    *(_QWORD *)(a2 + 32),
    1u,
    0,
    (__int64)&v13,
    0,
    1);
  v11 = v15;
  if ( v13 )
    sub_161E7C0((__int64)&v13, v13);
  return v11;
}
