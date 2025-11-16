// Function: sub_26C37A0
// Address: 0x26c37a0
//
void *__fastcall sub_26C37A0(__int64 a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // r12
  void *result; // rax

  v1 = *(_QWORD **)(a1 + 16);
  while ( v1 )
  {
    v2 = (unsigned __int64)v1;
    v1 = (_QWORD *)*v1;
    sub_C7D6A0(*(_QWORD *)(v2 + 40), 8LL * *(unsigned int *)(v2 + 56), 8);
    j_j___libc_free_0(v2);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
