// Function: sub_2FEC5A0
// Address: 0x2fec5a0
//
__int64 __fastcall sub_2FEC5A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 (*v3)(); // rax

  v2 = (*(_BYTE *)(a2 + 2) & 1) == 0 ? 2 : 6;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 9) )
    v2 |= 8u;
  v3 = *(__int64 (**)())(*(_QWORD *)a1 + 88LL);
  if ( v3 == sub_2FE2E30 )
    return v2;
  else
    return ((unsigned int (__fastcall *)(__int64, __int64))v3)(a1, a2) | v2;
}
