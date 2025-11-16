// Function: sub_6E53E0
// Address: 0x6e53e0
//
_BOOL8 __fastcall sub_6E53E0(char a1, unsigned int a2, _DWORD *a3)
{
  _BOOL8 result; // rax

  result = 1;
  if ( qword_4D03C50 && *(char *)(qword_4D03C50 + 18LL) < 0 )
  {
    result = sub_67D3C0((int *)a2, a1, a3);
    if ( result )
    {
      sub_6E50A0();
      return 0;
    }
  }
  return result;
}
