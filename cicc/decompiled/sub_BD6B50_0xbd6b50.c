// Function: sub_BD6B50
// Address: 0xbd6b50
//
int __fastcall sub_BD6B50(unsigned __int8 *a1, const char **a2)
{
  int result; // eax

  result = sub_BD6880(a1, a2);
  if ( !*a1 )
    return sub_B2DD70((__int64)a1);
  return result;
}
