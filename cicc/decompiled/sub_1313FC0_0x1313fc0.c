// Function: sub_1313FC0
// Address: 0x1313fc0
//
unsigned __int64 sub_1313FC0()
{
  unsigned __int64 result; // rax

  if ( (unsigned __int8)sub_130AF40((__int64)&unk_4F96AE0) || pthread_key_create(&key, (void (*)(void *))destr_function) )
    return 0;
  unk_4F96B58 = 1;
  result = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    return sub_1313D30(result, 0);
  return result;
}
