// Function: sub_85FC80
// Address: 0x85fc80
//
void sub_85FC80()
{
  __int64 v0; // rbx
  __int64 v1; // rbx

  v0 = unk_4F04C48;
  if ( unk_4F04C48 != -1 )
  {
    while ( 1 )
    {
      v1 = qword_4F04C68[0] + 776 * v0;
      if ( !v1 )
        break;
      if ( *(_BYTE *)(v1 + 4) == 9 && (*(_BYTE *)(v1 + 7) & 1) != 0 )
      {
        sub_85FC20(**(__int64 ****)(v1 + 408), 1);
        v0 = *(int *)(v1 + 552);
        if ( (_DWORD)v0 == -1 )
          return;
      }
      else
      {
        v0 = *(int *)(v1 + 552);
        if ( (_DWORD)v0 == -1 )
          return;
      }
    }
  }
}
