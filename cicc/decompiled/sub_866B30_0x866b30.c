// Function: sub_866B30
// Address: 0x866b30
//
__int64 sub_866B30()
{
  __int64 v0; // rax
  __int64 v1; // rax

  v0 = dword_4F04C64;
  do
  {
    v1 = qword_4F04C68[0] + 776 * v0;
    if ( !v1 )
      break;
    if ( *(_BYTE *)(v1 + 4) == 9 && (*(_BYTE *)(v1 + 6) & 6) == 0 )
      return 1;
    v0 = *(int *)(v1 + 552);
  }
  while ( (_DWORD)v0 != -1 );
  return 0;
}
