// Function: sub_1316F80
// Address: 0x1316f80
//
char *__fastcall sub_1316F80(__int64 a1, char *a2)
{
  void *v2; // rax

  v2 = memchr((const void *)(a1 + 78952), 0, 0x20u);
  return strncpy(a2, (const char *)(a1 + 78952), (size_t)v2 - a1 - 78951);
}
