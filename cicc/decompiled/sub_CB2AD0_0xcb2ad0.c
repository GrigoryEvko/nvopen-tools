// Function: sub_CB2AD0
// Address: 0xcb2ad0
//
void *__fastcall sub_CB2AD0(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // rsi
  size_t v5; // r13
  void *v6; // rdi
  void *result; // rax

  v3 = *(const void **)a1;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(void **)(a3 + 32);
  result = (void *)(*(_QWORD *)(a3 + 24) - (_QWORD)v6);
  if ( (unsigned __int64)result < v5 )
    return (void *)sub_CB6200(a3, v3, v5);
  if ( v5 )
  {
    result = memcpy(v6, v3, v5);
    *(_QWORD *)(a3 + 32) += v5;
  }
  return result;
}
