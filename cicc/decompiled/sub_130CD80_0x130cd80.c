// Function: sub_130CD80
// Address: 0x130cd80
//
bool __fastcall sub_130CD80(void *a1, size_t a2)
{
  return dword_4C6F0F0 || madvise(a1, a2, 4) != 0;
}
