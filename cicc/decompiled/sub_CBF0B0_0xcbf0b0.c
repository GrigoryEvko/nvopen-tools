// Function: sub_CBF0B0
// Address: 0xcbf0b0
//
unsigned __int64 __fastcall sub_CBF0B0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 i; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx

  for ( i = 0; i != 64; i += 16 )
  {
    v6 = *(_QWORD *)(a2 + i) ^ *(_QWORD *)(a1 + i);
    v7 = *(_QWORD *)(a2 + i + 8) ^ *(_QWORD *)(a1 + i + 8);
    a3 += ((v7 * (unsigned __int128)v6) >> 64) ^ (v7 * v6);
  }
  return (0x165667919E3779F9LL * ((a3 >> 37) ^ a3)) ^ ((0x165667919E3779F9LL * ((a3 >> 37) ^ a3)) >> 32);
}
