// Function: sub_641310
// Address: 0x641310
//
__int64 sub_641310()
{
  __int64 v0; // r8
  __int64 v1; // rcx
  __int64 i; // rax
  char v3; // dl

  LODWORD(v0) = dword_4F04C34;
  if ( dword_4F077BC )
  {
    if ( dword_4F04C34 )
    {
      v1 = 776LL * dword_4F04C34;
      for ( i = qword_4F04C68[0] + v1; *(_BYTE *)(i + 4) == 4; i = qword_4F04C68[0] + 776 * v0 )
      {
        if ( (*(_BYTE *)(i + 9) & 0x20) != 0 )
          break;
        v3 = *(_BYTE *)(i + 4);
        if ( v3 == 4 )
        {
          do
          {
            v3 = *(_BYTE *)(i - 772);
            i -= 776;
            if ( v3 != 4 )
              break;
          }
          while ( (*(_BYTE *)(i + 9) & 0x20) == 0 );
        }
        if ( v3 == 10 )
          break;
        v0 = *(int *)(qword_4F04C68[0] + v1 - 256);
        v1 = 776 * v0;
      }
    }
  }
  return (unsigned int)v0;
}
