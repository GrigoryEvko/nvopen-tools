// Function: sub_26C2A20
// Address: 0x26c2a20
//
void *__fastcall sub_26C2A20(__int64 a1)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rdi
  void *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  while ( v2 )
  {
    v3 = (unsigned __int64)v2;
    v2 = (_QWORD *)*v2;
    j_j___libc_free_0(v3);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
