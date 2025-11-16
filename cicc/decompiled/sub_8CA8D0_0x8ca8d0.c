// Function: sub_8CA8D0
// Address: 0x8ca8d0
//
__int64 __fastcall sub_8CA8D0(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v3 = a1;
    while ( a2 )
    {
      result = sub_8CA500(v3[2], a2[2]);
      v3 = (_QWORD *)*v3;
      a2 = (_QWORD *)*a2;
      if ( !v3 )
        goto LABEL_8;
    }
    do
    {
      result = sub_8CA0A0(v3[2], 1u);
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
  }
  else
  {
LABEL_8:
    while ( a2 )
    {
      result = sub_8CA0A0(a2[2], 1u);
      a2 = (_QWORD *)*a2;
    }
  }
  return result;
}
