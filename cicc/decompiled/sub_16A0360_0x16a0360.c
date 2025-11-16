// Function: sub_16A0360
// Address: 0x16a0360
//
__int64 *__fastcall sub_16A0360(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // rbx
  __int64 *result; // rax

  v2 = (__int64 *)sub_16982C0();
  result = (__int64 *)*a2;
  if ( (__int64 *)*a1 == v2 )
  {
    if ( v2 == result )
      return sub_16A0170(a1, a2);
  }
  else if ( v2 != result )
  {
    return sub_1698680(a1, a2);
  }
  if ( a1 != a2 )
  {
    sub_127D120(a1);
    if ( v2 == (__int64 *)*a2 )
      return sub_169C6E0(a1, (__int64)a2);
    else
      return (__int64 *)sub_16986C0(a1, a2);
  }
  return result;
}
