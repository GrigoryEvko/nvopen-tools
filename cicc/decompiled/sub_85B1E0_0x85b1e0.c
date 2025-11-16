// Function: sub_85B1E0
// Address: 0x85b1e0
//
__int64 sub_85B1E0()
{
  __int64 v0; // rax
  _BOOL4 v1; // ecx
  __int64 v2; // rax

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( !v0 )
    return 0;
  v1 = 0;
  while ( *(_BYTE *)(v0 + 4) != 9 )
  {
LABEL_4:
    v2 = *(int *)(v0 + 552);
    if ( (_DWORD)v2 != -1 )
    {
      v0 = qword_4F04C68[0] + 776 * v2;
      if ( v0 )
        continue;
    }
    return 0;
  }
  if ( !v1 )
  {
    if ( (*(_BYTE *)(v0 + 14) & 4) != 0 )
      v1 = (*(_BYTE *)(v0 + 6) & 2) != 0;
    goto LABEL_4;
  }
  return ((*(_BYTE *)(v0 + 6) >> 1) ^ 1) & 1;
}
