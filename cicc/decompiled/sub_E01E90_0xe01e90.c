// Function: sub_E01E90
// Address: 0xe01e90
//
char __fastcall sub_E01E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char result; // al

  if ( a2 == a3 )
    return 1;
  result = a3 == 0 || a2 == 0;
  if ( !result )
    return sub_E018D0(a2, a3, 0, a4, a5, a6);
  return result;
}
