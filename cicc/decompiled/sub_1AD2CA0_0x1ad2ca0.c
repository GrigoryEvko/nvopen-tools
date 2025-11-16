// Function: sub_1AD2CA0
// Address: 0x1ad2ca0
//
_QWORD *__fastcall sub_1AD2CA0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  _QWORD *v5; // r12
  _QWORD *result; // rax
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r8
  bool v9; // zf

  v2 = a1;
  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  if ( v3 <= 0 )
  {
LABEL_11:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return (_QWORD *)a2;
LABEL_19:
        v9 = sub_157ECB0(*(_QWORD *)(*v2 + 40LL)) == 0;
        result = v2;
        if ( v9 )
          return (_QWORD *)a2;
        return result;
      }
      v7 = sub_157ECB0(*(_QWORD *)(*v2 + 40LL));
      result = v2;
      if ( v7 )
        return result;
      ++v2;
    }
    v8 = sub_157ECB0(*(_QWORD *)(*v2 + 40LL));
    result = v2;
    if ( v8 )
      return result;
    ++v2;
    goto LABEL_19;
  }
  v5 = &a1[4 * v3];
  while ( 1 )
  {
    if ( sub_157ECB0(*(_QWORD *)(*v2 + 40LL)) )
      return v2;
    if ( sub_157ECB0(*(_QWORD *)(v2[1] + 40LL)) )
      return v2 + 1;
    if ( sub_157ECB0(*(_QWORD *)(v2[2] + 40LL)) )
      return v2 + 2;
    if ( sub_157ECB0(*(_QWORD *)(v2[3] + 40LL)) )
      return v2 + 3;
    v2 += 4;
    if ( v2 == v5 )
    {
      v4 = (a2 - (__int64)v2) >> 3;
      goto LABEL_11;
    }
  }
}
