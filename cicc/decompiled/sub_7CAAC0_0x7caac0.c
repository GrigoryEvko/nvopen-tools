// Function: sub_7CAAC0
// Address: 0x7caac0
//
int sub_7CAAC0()
{
  __int64 v0; // rdi
  int result; // eax
  __int64 i; // rbx
  __int64 v3; // rdi

  v0 = qword_4F17FE0;
  if ( qword_4F17FE0 )
  {
    result = dword_4F17FD8;
    if ( dword_4F17FD8 >= 0 )
    {
      for ( i = 112LL * dword_4F17FD8; ; i -= 112 )
      {
        v3 = i + v0;
        if ( (*(_BYTE *)(v3 + 88) & 0x40) != 0 )
        {
          *(_QWORD *)v3 = 0;
          if ( !i )
            break;
        }
        else
        {
          result = sub_720F70((FILE **)v3);
          if ( !i )
            break;
        }
        v0 = qword_4F17FE0;
      }
    }
  }
  dword_4F17FD8 = -1;
  qword_4F17FD0 = 0;
  return result;
}
