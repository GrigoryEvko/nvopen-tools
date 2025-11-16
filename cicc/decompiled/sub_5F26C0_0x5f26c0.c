// Function: sub_5F26C0
// Address: 0x5f26c0
//
__int64 sub_5F26C0()
{
  unsigned int v0; // r8d
  __int64 v1; // rax

  v0 = 0;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v1 + 4) == 6 )
    return *(_QWORD *)(*(_QWORD *)(v1 + 600) + 32LL) != 0;
  return v0;
}
