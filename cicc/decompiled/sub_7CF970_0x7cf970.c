// Function: sub_7CF970
// Address: 0x7cf970
//
__int64 sub_7CF970()
{
  __int64 result; // rax
  _QWORD *v1; // rdx

  result = unk_4F04C48;
  if ( unk_4F04C48 == -1 )
  {
    result = (unsigned int)dword_4F04C64;
    if ( dword_4F04C64 )
    {
      v1 = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 536);
      do
      {
        if ( *v1 )
          break;
        v1 -= 97;
        result = (unsigned int)(result - 1);
      }
      while ( (_DWORD)result );
    }
  }
  return result;
}
