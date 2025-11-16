// Function: sub_E9A370
// Address: 0xe9a370
//
__int64 __fastcall sub_E9A370(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax

  while ( 1 )
  {
    while ( 1 )
    {
      result = *a2;
      if ( (_BYTE)result != 3 )
        break;
      a2 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
    }
    if ( (unsigned __int8)result > 3u )
      break;
    if ( (_BYTE)result )
    {
      if ( (_BYTE)result == 2 )
      {
        result = *(_QWORD *)(*(_QWORD *)a1 + 64LL);
        if ( (void (*)())result != nullsub_340 )
          return ((__int64 (__fastcall *)(__int64, _QWORD))result)(a1, *((_QWORD *)a2 + 2));
      }
      return result;
    }
    sub_E9A370(a1, *((_QWORD *)a2 + 2));
    a2 = (unsigned __int8 *)*((_QWORD *)a2 + 3);
  }
  if ( (_BYTE)result == 4 )
    return (*(__int64 (__fastcall **)(unsigned __int8 *, __int64))(*((_QWORD *)a2 - 1) + 64LL))(a2 - 8, a1);
  return result;
}
