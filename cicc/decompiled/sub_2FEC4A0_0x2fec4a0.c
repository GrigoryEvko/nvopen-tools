// Function: sub_2FEC4A0
// Address: 0x2fec4a0
//
__int64 __fastcall sub_2FEC4A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 v7; // cx
  unsigned int v8; // r12d
  unsigned __int64 v9; // rax
  __int64 (*v10)(); // rax
  __int64 v12; // [rsp-8h] [rbp-48h]

  v7 = *(_WORD *)(a2 + 2);
  v8 = (v7 & 1) == 0 ? 1 : 5;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    if ( sub_B91C10(a2, 9) )
      v8 |= 8u;
    if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 6) )
      v8 |= 0x20u;
    v7 = *(_WORD *)(a2 + 2);
  }
  _BitScanReverse64(&v9, 1LL << (v7 >> 1));
  if ( sub_D305E0(*(_QWORD *)(a2 - 32), *(_QWORD *)(a2 + 8), 63 - (v9 ^ 0x3F), a3, a2, a4, 0, a5) )
    v8 |= 0x10u;
  v10 = *(__int64 (**)())(*(_QWORD *)a1 + 88LL);
  if ( v10 != sub_2FE2E30 )
    v8 |= ((__int64 (__fastcall *)(__int64, __int64, __int64))v10)(a1, a2, v12);
  return v8;
}
