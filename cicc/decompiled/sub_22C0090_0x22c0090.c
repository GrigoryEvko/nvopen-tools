// Function: sub_22C0090
// Address: 0x22c0090
//
void __fastcall sub_22C0090(unsigned __int8 *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  if ( (unsigned int)*a1 - 4 <= 1 )
  {
    if ( *((_DWORD *)a1 + 8) > 0x40u )
    {
      v2 = *((_QWORD *)a1 + 3);
      if ( v2 )
        j_j___libc_free_0_0(v2);
    }
    if ( *((_DWORD *)a1 + 4) > 0x40u )
    {
      v3 = *((_QWORD *)a1 + 1);
      if ( v3 )
        j_j___libc_free_0_0(v3);
    }
  }
}
