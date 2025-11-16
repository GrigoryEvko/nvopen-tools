// Function: sub_5D2E00
// Address: 0x5d2e00
//
__int64 __fastcall sub_5D2E00(__int64 a1, _QWORD *a2)
{
  if ( a2 )
    *a2 = 0;
  return ((*(_BYTE *)(a1 + 142) >> 6) ^ 1) & 1;
}
