// Function: sub_679A30
// Address: 0x679a30
//
__int64 *__fastcall sub_679A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  int i; // ebx
  __int64 v10; // rdx
  __int64 v11; // rcx

  result = (__int64 *)sub_7B8B50(a1, a2, a3, a4);
  if ( word_4F06418[0] == 25 )
  {
    sub_7B8B50(a1, a2, v5, v6);
    for ( i = 0; ; --i )
    {
      while ( 1 )
      {
        sub_7B8B50(a1, a2, v7, v8);
        result = (__int64 *)word_4F06418[0];
        if ( word_4F06418[0] == 26 )
          break;
        while ( (_WORD)result != 25 )
        {
          if ( (_WORD)result == 9 )
            return result;
          sub_7B8B50(a1, a2, v7, v8);
          result = (__int64 *)word_4F06418[0];
          if ( word_4F06418[0] == 26 )
            goto LABEL_8;
        }
        ++i;
      }
LABEL_8:
      if ( !i )
        break;
    }
    result = sub_679930((unsigned __int16)a1, 0, v7, v8);
    if ( word_4F06418[0] == 26 )
      return sub_679930((unsigned __int16)a1, 0, v10, v11);
  }
  return result;
}
