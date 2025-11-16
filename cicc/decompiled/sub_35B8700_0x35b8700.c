// Function: sub_35B8700
// Address: 0x35b8700
//
bool __fastcall sub_35B8700(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx

  v2 = *(_QWORD *)(**(_QWORD **)(a2 + 16) + 24LL * *(_QWORD *)(a2 + 8));
  v3 = *(_QWORD *)(**(_QWORD **)(a1 + 16) + 24LL * *(_QWORD *)(a1 + 8));
  return (*(_DWORD *)((v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v3 >> 1) & 3) > (*(_DWORD *)((v2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(v2 >> 1) & 3);
}
