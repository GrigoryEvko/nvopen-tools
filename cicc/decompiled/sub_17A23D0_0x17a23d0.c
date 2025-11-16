// Function: sub_17A23D0
// Address: 0x17a23d0
//
void __fastcall sub_17A23D0(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A89F0(a1, a2);
  else
    *a1 |= *a2;
}
