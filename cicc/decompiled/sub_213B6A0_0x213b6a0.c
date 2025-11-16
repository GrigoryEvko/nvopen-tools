// Function: sub_213B6A0
// Address: 0x213b6a0
//
__int64 __fastcall sub_213B6A0(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v4; // ebx
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // r10
  __int64 v10; // r11
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // r15
  const __m128i *v15; // r9
  __int128 v17; // [rsp-40h] [rbp-A0h]
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v20,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = (unsigned __int8)v21;
  v5 = v22;
  v6 = sub_2138AD0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = v6;
  v10 = v7;
  v20 = v8;
  if ( v8 )
  {
    v19 = v7;
    v18 = v6;
    sub_1623A60((__int64)&v20, v8, 2);
    v9 = v18;
    v10 = v19;
  }
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)(a2 + 96);
  v13 = (_QWORD *)a1[1];
  v21 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v17 + 1) = v10;
  *(_QWORD *)&v17 = v9;
  v14 = sub_1D257D0(
          v13,
          v4,
          v5,
          (__int64)&v20,
          *(_QWORD *)v11,
          *(_QWORD *)(v11 + 8),
          *(_OWORD *)(v11 + 40),
          *(_OWORD *)(v11 + 80),
          v17,
          *(unsigned __int8 *)(a2 + 88),
          v12,
          *(_QWORD *)(a2 + 104),
          2,
          0);
  sub_2013400((__int64)a1, a2, 1, v14, (__m128i *)1, v15);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v14;
}
