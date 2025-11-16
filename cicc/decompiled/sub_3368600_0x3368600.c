// Function: sub_3368600
// Address: 0x3368600
//
__int64 __fastcall sub_3368600(_WORD *a1)
{
  int v1; // eax

  v1 = (unsigned __int16)*a1;
  if ( (unsigned __int16)v1 <= 1u || (unsigned __int16)(*a1 - 504) <= 7u )
    BUG();
  return *(_QWORD *)&byte_444C4A0[16 * v1 - 16];
}
