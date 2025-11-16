// Function: sub_5C6254
// Address: 0x5c6254
//
__int64 (**sub_5C6254())(void)
{
  __int64 (**result)(void); // rax

  result = &_gmon_start__;
  if ( &_gmon_start__ )
    return (__int64 (**)(void))_gmon_start__();
  return result;
}
