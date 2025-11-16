// Function: sub_36DCAA0
// Address: 0x36dcaa0
//
__int64 __fastcall sub_36DCAA0(__int64 a1, __int64 a2)
{
  __int64 *v3; // r13
  __m128i v4; // xmm0
  unsigned __int16 *v5; // rax
  __int64 v6; // rsi
  unsigned __int16 v7; // ax
  unsigned __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  __int128 v15; // [rsp-10h] [rbp-80h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  __m128i v17; // [rsp+20h] [rbp-50h] BYREF
  __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  int v19; // [rsp+38h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 64);
  v4 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v17 = v4;
  v7 = *v5;
  v18 = v6;
  v8 = v7;
  v9 = v7;
  if ( v6 )
  {
    v16 = v7;
    sub_B96E90((__int64)&v18, v6, 1);
    v9 = v16;
  }
  *((_QWORD *)&v15 + 1) = 1;
  *(_QWORD *)&v15 = &v17;
  v19 = *(_DWORD *)(a2 + 72);
  v13 = sub_33E6B00(v3, 1564, (__int64)&v18, v9, 0, 1, v8, v15);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v13, v10, v11, v12);
  sub_3421DB0(v13);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  return 1;
}
