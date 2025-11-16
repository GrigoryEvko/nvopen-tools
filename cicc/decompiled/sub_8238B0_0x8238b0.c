// Function: sub_8238B0
// Address: 0x8238b0
//
void *__fastcall sub_8238B0(_QWORD *a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned __int64 v8; // r14
  void *result; // rax
  size_t v10; // [rsp+8h] [rbp-28h]

  v7 = a1[2];
  v8 = v7 + a3;
  if ( a1[1] < v7 + a3 )
  {
    v10 = a3;
    sub_823810(a1, v8, a3, a4, a5, a6);
    v7 = a1[2];
    a3 = v10;
  }
  result = memcpy((void *)(a1[4] + v7), a2, a3);
  a1[2] = v8;
  return result;
}
