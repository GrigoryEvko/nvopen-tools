// Function: sub_986FD0
// Address: 0x986fd0
//
char __fastcall sub_986FD0(__int64 a1, _QWORD *a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return (*a2 & *(_QWORD *)a1) != 0;
  else
    return sub_C446A0();
}
