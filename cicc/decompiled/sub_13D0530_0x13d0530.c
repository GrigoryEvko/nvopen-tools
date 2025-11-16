// Function: sub_13D0530
// Address: 0x13d0530
//
char __fastcall sub_13D0530(__int64 a1, _QWORD *a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return (*a2 & *(_QWORD *)a1) != 0;
  else
    return sub_16A59B0(a1, a2);
}
