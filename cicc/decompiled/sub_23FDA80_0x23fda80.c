// Function: sub_23FDA80
// Address: 0x23fda80
//
void *__fastcall sub_23FDA80(__int64 a1, __int64 a2)
{
  void *result; // rax
  __int64 v4; // rdi
  void *v5; // rax
  __int64 v6; // rdx

  result = (void *)sub_C7D6A0(*(_QWORD *)(a1 + 8), 8LL * *(unsigned int *)(a1 + 24), 8);
  v4 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v4;
  if ( (_DWORD)v4 )
  {
    v5 = (void *)sub_C7D670(8 * v4, 8);
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 8) = v5;
    *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
    *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
    return memcpy(v5, *(const void **)(a2 + 8), 8 * v6);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  return result;
}
