// Function: sub_1455820
// Address: 0x1455820
//
char __fastcall sub_1455820(__int64 a1, _QWORD *a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return *(_QWORD *)a1 == *a2;
  else
    return sub_16A5220(a1, a2);
}
