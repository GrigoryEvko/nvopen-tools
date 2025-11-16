// Function: sub_1F6C9A0
// Address: 0x1f6c9a0
//
void __fastcall sub_1F6C9A0(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A89F0(a1, a2);
  else
    *a1 |= *a2;
}
