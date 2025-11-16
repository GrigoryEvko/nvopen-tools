// Function: sub_130AA00
// Address: 0x130aa00
//
__int64 __fastcall sub_130AA00(__int64 a1, const char *a2)
{
  size_t v2; // rax

  v2 = strlen(a2);
  return syscall(1, 2, a2, v2);
}
