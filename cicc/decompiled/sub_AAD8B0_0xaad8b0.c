// Function: sub_AAD8B0
// Address: 0xaad8b0
//
char __fastcall sub_AAD8B0(__int64 a1, _QWORD *a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    return *(_QWORD *)a1 == *a2;
  else
    return sub_C43C50(a1, a2);
}
