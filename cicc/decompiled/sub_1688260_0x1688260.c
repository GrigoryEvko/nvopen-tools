// Function: sub_1688260
// Address: 0x1688260
//
void *__fastcall sub_1688260(__int64 a1, void **a2)
{
  size_t v2; // r12
  void *result; // rax

  v2 = *(_QWORD *)a1 - *(_QWORD *)(a1 + 8);
  result = memcpy(*a2, *(const void **)(a1 + 16), v2);
  *a2 = (char *)*a2 + v2;
  return result;
}
