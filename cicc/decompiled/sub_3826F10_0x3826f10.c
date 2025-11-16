// Function: sub_3826F10
// Address: 0x3826f10
//
void __fastcall sub_3826F10(
        __int64 a1,
        unsigned __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  unsigned __int64 v15; // r14
  __int128 v16; // rax
  __int64 *v17; // r14
  unsigned int v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+28h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v20 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v20, v9, 1);
  v10 = *(_QWORD **)(a1 + 8);
  v19 = *(_QWORD *)(a2 + 104);
  v11 = *(unsigned __int16 *)(a2 + 96);
  v18 = *(unsigned __int16 *)(a2 + 96);
  v21 = *(_DWORD *)(a2 + 72);
  v12 = sub_33E5B50(v10, v11, v19, 2, 0, a7, 1, 0);
  v14 = v13;
  v15 = v12;
  *(_QWORD *)&v16 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v20, v18, v19, 0, a3, 0);
  v17 = sub_33E6F00(
          *(_QWORD **)(a1 + 8),
          341,
          (__int64)&v20,
          *(_WORD *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          v15,
          v14,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          v16,
          v16);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v17, 0);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v17, 2);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
}
