// Function: sub_E45160
// Address: 0xe45160
//
void *__fastcall sub_E45160(__int64 *a1, __int64 a2)
{
  const char *v3; // rax
  size_t v4; // rdx
  void *v5; // rdi
  unsigned __int8 *v6; // rsi
  void *result; // rax
  size_t v8; // [rsp+8h] [rbp-18h]

  v3 = sub_BD5D20(*a1);
  v5 = *(void **)(a2 + 32);
  v6 = (unsigned __int8 *)v3;
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v5);
  if ( (unsigned __int64)result < v4 )
    return (void *)sub_CB6200(a2, v6, v4);
  if ( v4 )
  {
    v8 = v4;
    result = memcpy(v5, v6, v4);
    *(_QWORD *)(a2 + 32) += v8;
  }
  return result;
}
