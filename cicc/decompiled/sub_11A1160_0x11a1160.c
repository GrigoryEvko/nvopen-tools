// Function: sub_11A1160
// Address: 0x11a1160
//
__int64 __fastcall sub_11A1160(__int64 a1, __int64 *a2)
{
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    sub_C43B90((_QWORD *)a1, a2);
  else
    *(_QWORD *)a1 &= *a2;
  return a1;
}
