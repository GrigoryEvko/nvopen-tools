// Function: sub_39FAD80
// Address: 0x39fad80
//
void (__fastcall *sub_39FAD80())()
{
  void (__fastcall *result)(); // rax
  __int64 (__fastcall **v1)(); // rbx

  result = (void (__fastcall *)())off_496EDE0;
  if ( off_496EDE0 != (__int64 (__fastcall *)())-1LL )
  {
    v1 = &off_496EDE0;
    do
    {
      result();
      result = (void (__fastcall *)())*--v1;
    }
    while ( result != (void (__fastcall *)())-1LL );
  }
  return result;
}
