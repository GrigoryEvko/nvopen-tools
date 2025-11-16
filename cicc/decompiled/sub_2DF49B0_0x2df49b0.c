// Function: sub_2DF49B0
// Address: 0x2df49b0
//
__int64 __fastcall sub_2DF49B0(__int64 a1)
{
  __int64 v1; // rdx

  v1 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  return *(_QWORD *)v1 + 16LL * *(unsigned int *)(v1 + 12) + 8;
}
