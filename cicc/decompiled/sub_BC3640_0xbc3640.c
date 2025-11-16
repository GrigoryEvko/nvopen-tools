// Function: sub_BC3640
// Address: 0xbc3640
//
void *__fastcall sub_BC3640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  size_t v5; // rbx
  __int64 v6; // r13
  void *v7; // rdi
  void *result; // rax
  const void *v9; // rsi
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = -1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( a4 && !(unsigned __int8)sub_C93C90(a3, a4, 10, v10) )
    v5 = v10[0];
  v7 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(v6 + 8) <= v5 )
    v5 = *(_QWORD *)(v6 + 8);
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v7);
  v9 = *(const void **)v6;
  if ( (unsigned __int64)result < v5 )
    return (void *)sub_CB6200(a2, v9, v5);
  if ( v5 )
  {
    result = memcpy(v7, v9, v5);
    *(_QWORD *)(a2 + 32) += v5;
  }
  return result;
}
