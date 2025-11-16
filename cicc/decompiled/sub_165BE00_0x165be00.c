// Function: sub_165BE00
// Address: 0x165be00
//
__int64 __fastcall sub_165BE00(_QWORD *a1, unsigned int a2)
{
  return *(_QWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL)
                   + 24 * (a2 - (unsigned __int64)(*(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
}
