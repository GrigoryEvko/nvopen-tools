// Function: sub_155D460
// Address: 0x155d460
//
bool __fastcall sub_155D460(__int64 *a1, int a2)
{
  __int64 v3; // rdi
  bool result; // al

  v3 = *a1;
  if ( v3 )
  {
    result = sub_155D430(v3, a2);
    if ( result )
      return result;
    v3 = *a1;
  }
  return v3 == 0 && a2 == 0;
}
