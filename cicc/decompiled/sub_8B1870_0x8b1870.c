// Function: sub_8B1870
// Address: 0x8b1870
//
__int64 sub_8B1870()
{
  __int64 result; // rax
  __int64 v1; // rdi

  dword_4F601E0 = 1;
  result = dword_4D04734 - 1;
  if ( (unsigned int)result <= 1 )
  {
    do
    {
      dword_4F601C8 = 0;
      sub_8B17A0();
      v1 = qword_4F07288;
      nullsub_9();
      sub_5EB2E0(v1);
      result = (unsigned int)dword_4F601C8;
    }
    while ( dword_4F601C8 );
    dword_4F601E0 = 0;
  }
  else
  {
    dword_4F601E0 = 0;
  }
  return result;
}
