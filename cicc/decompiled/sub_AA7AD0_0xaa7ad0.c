// Function: sub_AA7AD0
// Address: 0xaa7ad0
//
_QWORD *__fastcall sub_AA7AD0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // [rsp+8h] [rbp-18h]

  if ( a2 == a1 + 48 )
  {
    result = (_QWORD *)sub_AA60B0(a1);
    if ( !result )
    {
      v3 = (_QWORD *)sub_22077B0(24);
      if ( v3 )
      {
        *v3 = 0;
        v3[2] = v3 + 1;
        v3[1] = (unsigned __int64)(v3 + 1) | 4;
      }
      v4 = v3;
      sub_AA7790(a1, (__int64)v3);
      return v4;
    }
  }
  else
  {
    if ( a2 )
      a2 -= 24;
    return sub_AA4580(a1, a2);
  }
  return result;
}
