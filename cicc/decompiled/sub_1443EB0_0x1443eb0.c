// Function: sub_1443EB0
// Address: 0x1443eb0
//
bool __fastcall sub_1443EB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  unsigned int v5; // r8d
  bool result; // al

  v4 = sub_157EBA0(a2);
  if ( v4 )
  {
    v5 = sub_15F4D60(v4);
    result = 0;
    if ( v5 > 1 )
      return result;
    v4 = sub_157EBA0(a2);
  }
  return a3 == sub_15F4DF0(v4, 0);
}
