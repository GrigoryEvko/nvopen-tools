// Function: sub_1727240
// Address: 0x1727240
//
void __fastcall sub_1727240(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A8F00(a1, a2);
  else
    *a1 ^= *a2;
}
