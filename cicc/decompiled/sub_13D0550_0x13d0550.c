// Function: sub_13D0550
// Address: 0x13d0550
//
char __fastcall sub_13D0550(__int64 a1, _QWORD *a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return (*(_QWORD *)a1 & ~*a2) == 0;
  else
    return sub_16A5A00();
}
