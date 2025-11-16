// Function: sub_10024C0
// Address: 0x10024c0
//
char __fastcall sub_10024C0(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) <= 0x40u )
    return (*a1 & ~*a2) == 0;
  else
    return sub_C446F0(a1, a2);
}
