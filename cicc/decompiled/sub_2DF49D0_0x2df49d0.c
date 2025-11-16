// Function: sub_2DF49D0
// Address: 0x2df49d0
//
__int64 __fastcall sub_2DF49D0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  return *(_QWORD *)v1 + 24LL * *(unsigned int *)(v1 + 12) + 64;
}
