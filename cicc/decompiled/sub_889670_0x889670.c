// Function: sub_889670
// Address: 0x889670
//
_BOOL8 __fastcall sub_889670(unsigned __int64 a1)
{
  unsigned __int64 i; // rdx
  unsigned int v2; // edx
  __int64 v3; // rax

  for ( i = a1 >> 3; ; LODWORD(i) = v2 + 1 )
  {
    v2 = *(_DWORD *)(qword_4F600F8 + 8) & i;
    v3 = *(_QWORD *)qword_4F600F8 + 16LL * v2;
    if ( a1 == *(_QWORD *)v3 )
      return *(_DWORD *)(v3 + 8) == 0;
    if ( !*(_QWORD *)v3 )
      break;
  }
  return 1;
}
