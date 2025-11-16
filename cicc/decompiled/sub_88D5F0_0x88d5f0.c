// Function: sub_88D5F0
// Address: 0x88d5f0
//
__int64 sub_88D5F0()
{
  unsigned int v0; // r8d
  __int64 i; // rax
  __int64 v2; // rax

  v0 = 0;
  for ( i = qword_4F04C68[0] + 776LL * dword_4F04C64; i; i = qword_4F04C68[0] + 776 * v2 )
  {
    if ( *(_BYTE *)(i + 4) == 9 && *(char *)(i + 9) >= 0 )
      v0 -= (*(_QWORD *)(i + 376) == 0) - 1;
    v2 = *(int *)(i + 552);
    if ( (_DWORD)v2 == -1 )
      break;
  }
  return v0;
}
