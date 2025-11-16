// Function: sub_728460
// Address: 0x728460
//
_QWORD *__fastcall sub_728460(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rcx

  if ( dword_4F07590 || (result = (_QWORD *)sub_8DBE70(a1), !(_DWORD)result) )
  {
    result = *(_QWORD **)(a2 + 360);
    if ( result )
    {
      v3 = 0;
      while ( result[1] != a1 )
      {
        v3 = result;
        if ( !*result )
          goto LABEL_10;
        result = (_QWORD *)*result;
      }
      if ( v3 )
      {
        *v3 = *result;
        *result = *(_QWORD *)(a2 + 360);
        *(_QWORD *)(a2 + 360) = result;
      }
    }
    else
    {
LABEL_10:
      result = (_QWORD *)sub_823970(16);
      result[1] = a1;
      *result = *(_QWORD *)(a2 + 360);
      *(_QWORD *)(a2 + 360) = result;
    }
  }
  return result;
}
