// Function: sub_96A970
// Address: 0x96a970
//
__int64 __fastcall sub_96A970(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // r13

  v2 = sub_C33340();
  result = *a2;
  if ( *a1 != v2 )
  {
    if ( v2 != result )
      return sub_C33E70(a1, a2);
    if ( a1 == a2 )
      return result;
    sub_C338F0(a1);
    v4 = *a2;
    goto LABEL_6;
  }
  if ( v2 == result )
    return sub_C3C9E0(a1, a2);
  if ( a1 != a2 )
  {
    v5 = (_QWORD *)a1[1];
    if ( !v5 )
      return sub_C33EB0(a1, a2);
    v6 = &v5[3 * *(v5 - 1)];
    if ( v5 != v6 )
    {
      do
      {
        v6 -= 3;
        if ( v2 == *v6 )
          sub_969EE0((__int64)v6);
        else
          sub_C338F0(v6);
      }
      while ( (_QWORD *)a1[1] != v6 );
    }
    j_j_j___libc_free_0_0(v6 - 1);
    v4 = *a2;
LABEL_6:
    if ( v2 != v4 )
      return sub_C33EB0(a1, a2);
    return sub_C3C790(a1, a2);
  }
  return result;
}
