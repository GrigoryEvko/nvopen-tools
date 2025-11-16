// Function: sub_38DDAF0
// Address: 0x38ddaf0
//
__int64 __fastcall sub_38DDAF0(__int64 a1, unsigned int *a2)
{
  __int64 result; // rax

  while ( 1 )
  {
    while ( 1 )
    {
      result = *a2;
      if ( (_DWORD)result != 3 )
        break;
      a2 = (unsigned int *)*((_QWORD *)a2 + 3);
    }
    if ( (unsigned int)result > 3 )
      break;
    if ( (_DWORD)result )
    {
      if ( (_DWORD)result == 2 )
      {
        result = *(_QWORD *)(*(_QWORD *)a1 + 56LL);
        if ( (void (*)())result != nullsub_1937 )
          return ((__int64 (__fastcall *)(__int64, _QWORD))result)(a1, *((_QWORD *)a2 + 3));
      }
      return result;
    }
    sub_38DDAF0(a1, *((_QWORD *)a2 + 3));
    a2 = (unsigned int *)*((_QWORD *)a2 + 4);
  }
  if ( (_DWORD)result == 4 )
    return (*(__int64 (__fastcall **)(unsigned int *, __int64))(*((_QWORD *)a2 - 1) + 56LL))(a2 - 2, a1);
  return result;
}
