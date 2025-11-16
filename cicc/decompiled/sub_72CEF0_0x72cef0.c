// Function: sub_72CEF0
// Address: 0x72cef0
//
__int64 sub_72CEF0()
{
  __int64 v0; // rax
  unsigned int v1; // r8d
  char v2; // dl

  v0 = qword_4F07AA0;
  v1 = 0;
  if ( qword_4F07AA0 )
  {
    while ( 1 )
    {
      v2 = *(_BYTE *)(v0 + 140);
      if ( v2 != 12 )
        break;
      v0 = *(_QWORD *)(v0 + 160);
    }
    return v2 != 0;
  }
  return v1;
}
