// Function: sub_2537BA0
// Address: 0x2537ba0
//
_QWORD *__fastcall sub_2537BA0(_QWORD *a1, __int64 a2, unsigned __int8 (__fastcall *a3)(__int64, _QWORD), __int64 a4)
{
  __int64 v5; // r14
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  _QWORD *result; // rax
  unsigned __int8 v11; // r8
  unsigned __int8 v12; // r8
  bool v13; // zf

  v5 = (a2 - (__int64)a1) >> 5;
  v7 = (a2 - (__int64)a1) >> 3;
  v8 = a1;
  if ( v5 <= 0 )
  {
LABEL_11:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (_QWORD *)a2;
LABEL_19:
        v13 = a3(a4, *v8) == 0;
        result = v8;
        if ( !v13 )
          return (_QWORD *)a2;
        return result;
      }
      v11 = a3(a4, *v8);
      result = v8;
      if ( !v11 )
        return result;
      ++v8;
    }
    v12 = a3(a4, *v8);
    result = v8;
    if ( !v12 )
      return result;
    ++v8;
    goto LABEL_19;
  }
  v9 = &a1[4 * v5];
  while ( 1 )
  {
    if ( !a3(a4, *v8) )
      return v8;
    if ( !a3(a4, v8[1]) )
      return v8 + 1;
    if ( !a3(a4, v8[2]) )
      return v8 + 2;
    if ( !a3(a4, v8[3]) )
      return v8 + 3;
    v8 += 4;
    if ( v8 == v9 )
    {
      v7 = (a2 - (__int64)v8) >> 3;
      goto LABEL_11;
    }
  }
}
