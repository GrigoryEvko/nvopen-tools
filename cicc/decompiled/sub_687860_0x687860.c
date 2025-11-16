// Function: sub_687860
// Address: 0x687860
//
__int64 sub_687860()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = (unsigned int)dword_4F04C58;
  if ( dword_4F04C58 == -1 )
  {
    result = dword_4F04C64;
    if ( unk_4F04C38 )
    {
      if ( dword_4F04C64 == -1 )
      {
        return 0;
      }
      else
      {
        while ( 1 )
        {
          v1 = qword_4F04C68[0] + 776 * result;
          result = *(unsigned int *)(v1 + 400);
          if ( (_DWORD)result != -1 )
            break;
          if ( *(_BYTE *)(v1 + 4) == 9 )
            result = *(int *)(v1 + 572);
          else
            result = *(int *)(v1 + 552);
          if ( (_DWORD)result == -1 )
            return 0;
        }
      }
    }
  }
  return result;
}
