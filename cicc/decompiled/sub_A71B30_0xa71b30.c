// Function: sub_A71B30
// Address: 0xa71b30
//
bool __fastcall sub_A71B30(__int64 *a1, int a2)
{
  __int64 v3; // rdi
  bool result; // al

  v3 = *a1;
  if ( v3 )
  {
    result = sub_A71B00(v3, a2);
    if ( result )
      return result;
    v3 = *a1;
  }
  return v3 == 0 && a2 == 0;
}
