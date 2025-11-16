// Function: sub_13D0200
// Address: 0x13d0200
//
bool __fastcall sub_13D0200(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdx

  v2 = *a1;
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    v2 = *(_QWORD *)(v2 + 8LL * (a2 >> 6));
  return ((1LL << a2) & v2) != 0;
}
