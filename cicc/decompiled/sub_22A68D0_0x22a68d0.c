// Function: sub_22A68D0
// Address: 0x22a68d0
//
void *__fastcall sub_22A68D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  size_t v6; // rbx
  void *v7; // rdi
  void *result; // rax
  unsigned __int8 *v9; // rsi
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = -1;
  if ( a4 && !sub_C93C90(a3, a4, 0xAu, v10) )
    v6 = v10[0];
  v7 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a1 + 16) <= v6 )
    v6 = *(_QWORD *)(a1 + 16);
  result = (void *)(*(_QWORD *)(a2 + 24) - (_QWORD)v7);
  v9 = *(unsigned __int8 **)(a1 + 8);
  if ( (unsigned __int64)result < v6 )
    return (void *)sub_CB6200(a2, v9, v6);
  if ( v6 )
  {
    result = memcpy(v7, v9, v6);
    *(_QWORD *)(a2 + 32) += v6;
  }
  return result;
}
