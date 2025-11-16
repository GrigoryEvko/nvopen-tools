// Function: sub_2DF8300
// Address: 0x2df8300
//
bool __fastcall sub_2DF8300(__int64 *a1, __int64 a2)
{
  return (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) > (*(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(*a1 >> 1) & 3);
}
