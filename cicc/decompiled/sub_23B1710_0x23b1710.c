// Function: sub_23B1710
// Address: 0x23b1710
//
void *__fastcall sub_23B1710(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  size_t v5; // rbx
  __int64 v6; // r13
  void *v7; // rdi
  void *result; // rax
  unsigned __int8 *v9; // rsi
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = -1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( a4 && !sub_C93C90(a3, a4, 0xAu, v10) )
    v5 = v10[0];
  v7 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(v6 + 8) <= v5 )
    v5 = *(_QWORD *)(v6 + 8);
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v7);
  v9 = *(unsigned __int8 **)v6;
  if ( (unsigned __int64)result < v5 )
    return (void *)sub_CB6200(a2, v9, v5);
  if ( v5 )
  {
    result = memcpy(v7, v9, v5);
    *(_QWORD *)(a2 + 32) += v5;
  }
  return result;
}
