// Function: sub_1F6C9C0
// Address: 0x1f6c9c0
//
void __fastcall sub_1F6C9C0(__int64 *a1, __int64 *a2)
{
  if ( *((_DWORD *)a1 + 2) > 0x40u )
    sub_16A8890(a1, a2);
  else
    *a1 &= *a2;
}
