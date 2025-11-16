// Function: sub_C65780
// Address: 0xc65780
//
void *__fastcall sub_C65780(__int64 a1)
{
  void *result; // rax

  memset(*(void **)a1, 0, 8LL * *(unsigned int *)(a1 + 8));
  result = *(void **)a1;
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) = -1;
  *(_DWORD *)(a1 + 12) = 0;
  return result;
}
