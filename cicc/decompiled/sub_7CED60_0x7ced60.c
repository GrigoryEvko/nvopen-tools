// Function: sub_7CED60
// Address: 0x7ced60
//
_QWORD *__fastcall sub_7CED60(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  _QWORD *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rax

  result = (_QWORD *)*a3;
  if ( *a3 )
  {
    while ( 1 )
    {
      v6 = result[1];
      if ( v6 == a1 )
        break;
      if ( a1 )
      {
        if ( v6 )
        {
          if ( dword_4F07588 )
          {
            v5 = *(_QWORD *)(v6 + 32);
            if ( *(_QWORD *)(a1 + 32) == v5 )
            {
              if ( v5 )
                break;
            }
          }
        }
      }
      result = (_QWORD *)*result;
      if ( !result )
        goto LABEL_11;
    }
  }
  else
  {
LABEL_11:
    v7 = (_QWORD *)sub_8784C0();
    v7[1] = a1;
    *v7 = *a3;
    *a3 = v7;
    return sub_7CECA0(a1, a2);
  }
  return result;
}
