// Function: sub_16E57C0
// Address: 0x16e57c0
//
void *__fastcall sub_16E57C0(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v3; // rsi
  size_t v5; // r13
  void *v6; // rdi
  void *result; // rax

  v3 = *(const char **)a1;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(void **)(a3 + 24);
  result = (void *)(*(_QWORD *)(a3 + 16) - (_QWORD)v6);
  if ( (unsigned __int64)result < v5 )
    return (void *)sub_16E7EE0(a3, v3, v5);
  if ( v5 )
  {
    result = memcpy(v6, v3, v5);
    *(_QWORD *)(a3 + 24) += v5;
  }
  return result;
}
