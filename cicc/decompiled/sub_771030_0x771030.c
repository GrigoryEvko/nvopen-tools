// Function: sub_771030
// Address: 0x771030
//
_QWORD *__fastcall sub_771030(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  result = (_QWORD *)*a1;
  if ( *a1 )
  {
    do
    {
      v4 = result[5];
      if ( v4 == a2 )
        break;
      if ( a2 )
      {
        if ( v4 )
        {
          if ( dword_4F07588 )
          {
            v3 = *(_QWORD *)(v4 + 32);
            if ( *(_QWORD *)(a2 + 32) == v3 )
            {
              if ( v3 )
                break;
            }
          }
        }
      }
      result = (_QWORD *)*result;
    }
    while ( result );
  }
  return result;
}
