// Function: sub_2AA8260
// Address: 0x2aa8260
//
_QWORD *__fastcall sub_2AA8260(_QWORD *a1, __int64 a2, unsigned __int8 (__fastcall *a3)(_QWORD))
{
  __int64 v4; // r13
  _QWORD *v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // r13
  _QWORD *result; // rax
  unsigned __int8 v9; // r8
  unsigned __int8 v10; // r8
  bool v11; // zf

  v4 = (a2 - (__int64)a1) >> 5;
  v5 = a1;
  v6 = (a2 - (__int64)a1) >> 3;
  if ( v4 <= 0 )
  {
LABEL_11:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return (_QWORD *)a2;
LABEL_19:
        v11 = a3(*v5) == 0;
        result = v5;
        if ( !v11 )
          return (_QWORD *)a2;
        return result;
      }
      v9 = a3(*v5);
      result = v5;
      if ( !v9 )
        return result;
      ++v5;
    }
    v10 = a3(*v5);
    result = v5;
    if ( !v10 )
      return result;
    ++v5;
    goto LABEL_19;
  }
  v7 = &a1[4 * v4];
  while ( 1 )
  {
    if ( !a3(*v5) )
      return v5;
    if ( !a3(v5[1]) )
      return v5 + 1;
    if ( !a3(v5[2]) )
      return v5 + 2;
    if ( !a3(v5[3]) )
      return v5 + 3;
    v5 += 4;
    if ( v5 == v7 )
    {
      v6 = (a2 - (__int64)v5) >> 3;
      goto LABEL_11;
    }
  }
}
