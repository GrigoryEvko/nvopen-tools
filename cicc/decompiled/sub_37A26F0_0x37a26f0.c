// Function: sub_37A26F0
// Address: 0x37a26f0
//
__m128i *__fastcall sub_37A26F0(__int64 **a1, unsigned __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // r14
  unsigned int v9; // edx
  unsigned __int64 v10; // r15
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v12; // rax
  unsigned __int16 v13; // si
  __int64 v14; // r8
  __int64 *v15; // rax
  __int64 v16; // r8
  unsigned int v17; // ecx
  __int64 v18; // rax
  __m128i *v19; // r14
  __int64 v21; // rdx
  __int128 v22; // [rsp-40h] [rbp-B0h]
  __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  int v24; // [rsp+18h] [rbp-58h]
  _BYTE v25[8]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int16 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v23 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v23, v3, 1);
  v24 = *(_DWORD *)(a2 + 72);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v4 + 168);
  v6 = sub_379AB60((__int64)a1, *(_QWORD *)(v4 + 160), v5);
  v7 = (__int64)*a1;
  v8 = v6;
  v10 = v9 | v5 & 0xFFFFFFFF00000000LL;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**a1 + 592);
  v12 = *(__int16 **)(a2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v15 = a1[1];
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v25, v7, v15[8], v13, v14);
    v16 = v27;
    v17 = v26;
  }
  else
  {
    v17 = v11(v7, v15[8], v13, v14);
    v16 = v21;
  }
  v18 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v22 + 1) = v10;
  *(_QWORD *)&v22 = v8;
  v19 = sub_33E8960(
          a1[1],
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (*(_BYTE *)(a2 + 33) >> 2) & 3,
          v17,
          v16,
          (__int64)&v23,
          *(_OWORD *)v18,
          *(_QWORD *)(v18 + 40),
          *(_QWORD *)(v18 + 48),
          *(_OWORD *)(v18 + 80),
          *(_OWORD *)(v18 + 120),
          v22,
          *(_OWORD *)(v18 + 200),
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v19, 1);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v19;
}
