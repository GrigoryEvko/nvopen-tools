// Function: sub_274B030
// Address: 0x274b030
//
__int64 __fastcall sub_274B030(unsigned __int8 *a1, unsigned int a2, char a3, char a4)
{
  __int64 result; // rax

  if ( a2 > 0x19 || (result = 1LL << a2, ((1LL << a2) & 0x202A000) == 0) )
    BUG();
  if ( *a1 > 0x1Cu )
  {
    if ( a3 )
      result = sub_B44850(a1, 1);
    if ( a4 )
      return sub_B447F0(a1, 1);
  }
  return result;
}
