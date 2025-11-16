// Function: sub_36EB890
// Address: 0x36eb890
//
void __fastcall sub_36EB890(__int64 a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rax
  _DWORD *v7; // rax
  __int64 v8; // r9
  __int64 v9; // rsi
  _BOOL4 v10; // r13d
  int v11; // r13d
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // [rsp+0h] [rbp-50h] BYREF
  int v20; // [rsp+8h] [rbp-48h]
  _OWORD v21[4]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(const __m128i **)(a2 + 40);
  v21[0] = _mm_loadu_si128(v4 + 5);
  v5 = *(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL);
  v21[1] = _mm_loadu_si128(v4);
  v6 = sub_2E79000(v5);
  v7 = sub_AE2980(v6, 3u);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = v7[1] == 64;
  v19 = v9;
  v11 = v10 + 2866;
  if ( v9 )
    sub_B96E90((__int64)&v19, v9, 1);
  v12 = *(_QWORD **)(a1 + 64);
  v13 = *(_QWORD *)(a2 + 48);
  v14 = *(unsigned int *)(a2 + 68);
  v20 = *(_DWORD *)(a2 + 72);
  v15 = sub_33E66D0(v12, v11, (__int64)&v19, v13, v14, v8, (unsigned __int64 *)v21, 2);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v15, v16, v17, v18);
  sub_3421DB0(v15);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
}
