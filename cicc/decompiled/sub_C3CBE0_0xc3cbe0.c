// Function: sub_C3CBE0
// Address: 0xc3cbe0
//
__int64 *__fastcall sub_C3CBE0(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // rbx
  __int64 *result; // rax

  v2 = (__int64 *)sub_C33340();
  result = (__int64 *)*a2;
  if ( (__int64 *)*a1 == v2 )
  {
    if ( v2 == result )
      return sub_C3C9E0(a1, a2);
    if ( a1 != a2 )
    {
      sub_969EE0((__int64)a1);
LABEL_6:
      if ( v2 == (__int64 *)*a2 )
        return sub_C3C790(a1, (_QWORD **)a2);
      else
        return (__int64 *)sub_C33EB0(a1, a2);
    }
  }
  else
  {
    if ( v2 != result )
      return sub_C33E70(a1, a2);
    if ( a1 != a2 )
    {
      sub_C338F0((__int64)a1);
      goto LABEL_6;
    }
  }
  return result;
}
