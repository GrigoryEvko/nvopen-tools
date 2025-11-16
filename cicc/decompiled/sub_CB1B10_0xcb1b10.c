// Function: sub_CB1B10
// Address: 0xcb1b10
//
void *__fastcall sub_CB1B10(__int64 a1, const void *a2, size_t a3)
{
  __int64 v3; // r13
  void *v4; // rdi
  void *result; // rax

  v3 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 80) += a3;
  v4 = *(void **)(v3 + 32);
  result = (void *)(*(_QWORD *)(v3 + 24) - (_QWORD)v4);
  if ( (unsigned __int64)result < a3 )
    return (void *)sub_CB6200(v3, a2, a3);
  if ( a3 )
  {
    result = memcpy(v4, a2, a3);
    *(_QWORD *)(v3 + 32) += a3;
  }
  return result;
}
