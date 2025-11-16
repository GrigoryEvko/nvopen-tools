// Function: sub_1718FB0
// Address: 0x1718fb0
//
void __fastcall sub_1718FB0(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A8890(a1, a2);
  else
    *a1 &= *a2;
}
