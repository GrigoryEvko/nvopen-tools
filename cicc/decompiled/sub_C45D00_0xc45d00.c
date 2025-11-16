// Function: sub_C45D00
// Address: 0xc45d00
//
void *__fastcall sub_C45D00(_QWORD *a1, __int64 a2, unsigned int a3)
{
  void *result; // rax

  *a1 = a2;
  if ( a3 > 1 )
    return memset(a1 + 1, 0, 8LL * (a3 - 1));
  return result;
}
