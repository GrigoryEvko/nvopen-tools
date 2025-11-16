// Function: sub_85B130
// Address: 0x85b130
//
__int64 sub_85B130()
{
  int v0; // edx
  __int64 v1; // r8
  __int64 v2; // rax
  __int64 v3; // rax

  v0 = unk_4F04C48;
  if ( dword_4F04C44 >= unk_4F04C48 )
    v0 = dword_4F04C44;
  v1 = qword_4F04C68[0] - 776LL;
  if ( v0 != -1 )
  {
    v1 = qword_4F04C68[0] + 776LL * v0;
    v2 = v1;
    if ( v1 )
    {
      do
      {
        if ( *(_BYTE *)(v2 + 4) == 8 || (*(_DWORD *)(v2 + 4) & 0x200FF) == 0x20009 )
          v0 = 1594008481 * ((v2 - qword_4F04C68[0]) >> 3);
        v3 = *(int *)(v2 + 552);
        if ( (_DWORD)v3 == -1 )
          break;
        v2 = qword_4F04C68[0] + 776 * v3;
      }
      while ( v2 );
      return qword_4F04C68[0] + 776LL * v0;
    }
  }
  return v1;
}
