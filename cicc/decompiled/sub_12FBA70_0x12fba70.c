// Function: sub_12FBA70
// Address: 0x12fba70
//
unsigned __int64 __fastcall sub_12FBA70(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  if ( (a1 & 0x7FFF800000000000LL) == 0x7FFF000000000000LL && a2 | a1 & 0x7FFFFFFFFFFFLL )
    sub_12F9B70(16);
  *(_QWORD *)(a3 + 8) = a2 << 16;
  *(_BYTE *)a3 = a1 < 0;
  *(_QWORD *)(a3 + 16) = HIWORD(a2) | (a1 << 16);
  *(_BYTE *)a3 &= 1u;
  return (unsigned __int64)a1 >> 63;
}
