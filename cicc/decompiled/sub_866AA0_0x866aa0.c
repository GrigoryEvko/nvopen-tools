// Function: sub_866AA0
// Address: 0x866aa0
//
__int64 sub_866AA0()
{
  __int64 v0; // rax
  __int64 v1; // rax

  v0 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( dword_4F04C44 == -1 )
  {
    if ( (*(_BYTE *)(v0 + 6) & 2) != 0 )
      goto LABEL_6;
    return 0;
  }
  else if ( v0 )
  {
LABEL_6:
    while ( *(_BYTE *)(v0 + 4) != 9 || (*(_BYTE *)(v0 + 14) & 4) == 0 || (*(_BYTE *)(v0 + 6) & 2) == 0 )
    {
      v1 = *(int *)(v0 + 552);
      if ( (_DWORD)v1 != -1 )
      {
        v0 = qword_4F04C68[0] + 776 * v1;
        if ( v0 )
          continue;
      }
      return 0;
    }
    return 1;
  }
  else
  {
    return 0;
  }
}
