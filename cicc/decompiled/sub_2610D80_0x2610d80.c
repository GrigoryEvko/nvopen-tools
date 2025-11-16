// Function: sub_2610D80
// Address: 0x2610d80
//
void *__fastcall sub_2610D80(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v5; // rax
  size_t v6; // rdx
  void *v7; // rdi
  unsigned __int8 *v8; // rsi
  void *result; // rax
  size_t v10; // [rsp+8h] [rbp-18h]

  v5 = a3(a4, "InlineAdvisorAnalysisPrinterPass]", 32);
  v7 = *(void **)(a2 + 32);
  v8 = (unsigned __int8 *)v5;
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v7);
  if ( (unsigned __int64)result < v6 )
    return (void *)sub_CB6200(a2, v8, v6);
  if ( v6 )
  {
    v10 = v6;
    result = memcpy(v7, v8, v6);
    *(_QWORD *)(a2 + 32) += v10;
  }
  return result;
}
