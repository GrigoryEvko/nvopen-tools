// Function: sub_2208E60
// Address: 0x2208e60
//
__int64 sub_2208E60()
{
  __int64 result; // rax

  if ( &_pthread_key_create )
  {
    pthread_once(&dword_4FD4F3C, sub_2208D10);
    return unk_4FD4F40;
  }
  else
  {
    result = unk_4FD4F40;
    if ( !unk_4FD4F40 )
    {
      sub_2208D10();
      return unk_4FD4F40;
    }
  }
  return result;
}
