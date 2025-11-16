// Function: sub_11A11A0
// Address: 0x11a11a0
//
__int64 __fastcall sub_11A11A0(__int64 a1, __int64 *a2)
{
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    sub_C43BD0((_QWORD *)a1, a2);
  else
    *(_QWORD *)a1 |= *a2;
  return a1;
}
