// Function: sub_16A7020
// Address: 0x16a7020
//
void *__fastcall sub_16A7020(_QWORD *a1, __int64 a2, unsigned int a3)
{
  void *result; // rax

  *a1 = a2;
  if ( a3 > 1 )
    return memset(a1 + 1, 0, 8LL * (a3 - 1));
  return result;
}
