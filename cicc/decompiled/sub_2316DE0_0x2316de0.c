// Function: sub_2316DE0
// Address: 0x2316de0
//
void *__fastcall sub_2316DE0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v7; // rax
  size_t v8; // rdx
  void *v9; // rdi
  unsigned __int8 *v10; // rsi
  void *result; // rax
  size_t v12; // [rsp+8h] [rbp-38h]
  char *v13; // [rsp+10h] [rbp-30h] BYREF
  __int64 v14; // [rsp+18h] [rbp-28h]

  v13 = "llvm::LazyCallGraphDOTPrinterPass]";
  v14 = 33;
  sub_95CB50((const void **)&v13, "llvm::", 6u);
  v7 = a3(a4, v13, v14);
  v9 = *(void **)(a2 + 32);
  v10 = (unsigned __int8 *)v7;
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v9);
  if ( (unsigned __int64)result < v8 )
    return (void *)sub_CB6200(a2, v10, v8);
  if ( v8 )
  {
    v12 = v8;
    result = memcpy(v9, v10, v8);
    *(_QWORD *)(a2 + 32) += v12;
  }
  return result;
}
