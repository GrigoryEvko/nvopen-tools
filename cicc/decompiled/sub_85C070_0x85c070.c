// Function: sub_85C070
// Address: 0x85c070
//
__int64 __fastcall sub_85C070(_QWORD *a1)
{
  __int64 result; // rax
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v4; // r13

  result = 1;
  if ( !a1[14] && !a1[13] )
  {
    v2 = (_QWORD *)a1[20];
    if ( v2 )
    {
      while ( !v2[14] && !v2[13] )
      {
        v3 = (_QWORD *)v2[20];
        if ( v3 )
        {
          while ( !v3[14] && !v3[13] )
          {
            v4 = (_QWORD *)v3[20];
            if ( v4 )
            {
              while ( !(unsigned int)sub_85C070(v4) )
              {
                v4 = (_QWORD *)*v4;
                if ( !v4 )
                  goto LABEL_15;
              }
              return 1;
            }
LABEL_15:
            v3 = (_QWORD *)*v3;
            if ( !v3 )
              goto LABEL_16;
          }
          return 1;
        }
LABEL_16:
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          return 0;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
