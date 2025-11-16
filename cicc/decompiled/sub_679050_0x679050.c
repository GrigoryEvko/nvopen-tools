// Function: sub_679050
// Address: 0x679050
//
__int64 __fastcall sub_679050(_QWORD ***a1)
{
  _QWORD **v1; // r12
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  __int64 result; // rax

  if ( a1 )
  {
    v1 = *a1;
    if ( *a1 )
    {
      v2 = *v1;
      if ( *v1 )
      {
        v3 = (_QWORD *)*v2;
        if ( *v2 )
        {
          sub_679050(*v3);
          result = qword_4CFDE88;
          *v3 = qword_4CFDE88;
        }
        else
        {
          v3 = (_QWORD *)qword_4CFDE88;
        }
        *v2 = v3;
      }
      else
      {
        v2 = (_QWORD *)qword_4CFDE88;
      }
      *v1 = v2;
    }
    else
    {
      v1 = (_QWORD **)qword_4CFDE88;
    }
    *a1 = v1;
    qword_4CFDE88 = (__int64)a1;
  }
  return result;
}
