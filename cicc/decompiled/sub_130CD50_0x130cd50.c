// Function: sub_130CD50
// Address: 0x130cd50
//
bool __fastcall sub_130CD50(void *a1, size_t a2)
{
  return !byte_4C6F0F4 || madvise(a1, a2, 8) != 0;
}
