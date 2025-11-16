// Function: sub_7ABED0
// Address: 0x7abed0
//
__int64 __fastcall sub_7ABED0(char *a1)
{
  unsigned int v1; // r12d
  __int64 i; // rbx

  if ( dword_4F17FD8 < 0 )
    return 0;
  v1 = 0;
  for ( i = 112LL * dword_4F17FD8; ; i -= 112 )
  {
    v1 += !sub_722E50(*(char **)(qword_4F17FE0 + i + 16), a1, 0, 0, 0);
    if ( !i )
      break;
  }
  return v1;
}
