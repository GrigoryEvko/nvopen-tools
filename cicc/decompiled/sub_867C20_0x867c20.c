// Function: sub_867C20
// Address: 0x867c20
//
__int64 sub_867C20()
{
  unsigned int v0; // r8d
  __int64 v2; // rax

  v0 = dword_4F04C58;
  if ( dword_4F04C58 != -1 )
    return v0;
  if ( !dword_4F04C38 )
    return v0;
  v0 = dword_4F04C64;
  if ( dword_4F04C64 == -1 )
    return v0;
  while ( 1 )
  {
    v2 = qword_4F04C68[0] + 776LL * (int)v0;
    if ( *(_BYTE *)(v2 + 4) == 17 )
      break;
    v0 = *(_DWORD *)(v2 + 552);
    if ( v0 == -1 )
      return v0;
  }
  return v0;
}
