// Function: sub_16E4B40
// Address: 0x16e4b40
//
void *__fastcall sub_16E4B40(__int64 a1, const char *a2, size_t a3)
{
  __int64 v3; // r13
  void *v4; // rdi
  void *result; // rax

  v3 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 80) += a3;
  v4 = *(void **)(v3 + 24);
  result = (void *)(*(_QWORD *)(v3 + 16) - (_QWORD)v4);
  if ( (unsigned __int64)result < a3 )
    return (void *)sub_16E7EE0(v3, a2);
  if ( a3 )
  {
    result = memcpy(v4, a2, a3);
    *(_QWORD *)(v3 + 24) += a3;
  }
  return result;
}
