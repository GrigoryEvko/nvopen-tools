// Function: sub_1688A60
// Address: 0x1688a60
//
size_t sub_1688A60(__int64 a1, const char *a2, ...)
{
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  va_start(va, a2);
  return sub_1688930(a1, a2, (__m128i *)va);
}
