// Function: sub_1688630
// Address: 0x1688630
//
size_t sub_1688630(__int64 *a1, const char *a2, ...)
{
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  va_start(va, a2);
  return sub_1688540(a1, a2, (__m128i *)va);
}
