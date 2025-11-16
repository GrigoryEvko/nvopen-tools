// Function: sub_1D13530
// Address: 0x1d13530
//
void __fastcall sub_1D13530(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A8890(a1, a2);
  else
    *a1 &= *a2;
}
