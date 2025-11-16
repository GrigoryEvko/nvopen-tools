// Function: sub_2537880
// Address: 0x2537880
//
__int64 *__fastcall sub_2537880(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 *result; // rax
  char v9; // r8
  char v10; // r8
  bool v11; // zf

  v4 = (a2 - (__int64)a1) >> 6;
  v5 = a1;
  v6 = (a2 - (__int64)a1) >> 4;
  if ( v4 <= 0 )
  {
LABEL_11:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return (__int64 *)a2;
LABEL_19:
        v11 = (unsigned __int8)sub_250C180(*v5, *a3) == 0;
        result = v5;
        if ( !v11 )
          return (__int64 *)a2;
        return result;
      }
      v9 = sub_250C180(*v5, *a3);
      result = v5;
      if ( !v9 )
        return result;
      v5 += 2;
    }
    v10 = sub_250C180(*v5, *a3);
    result = v5;
    if ( !v10 )
      return result;
    v5 += 2;
    goto LABEL_19;
  }
  v7 = &a1[8 * v4];
  while ( 1 )
  {
    if ( !(unsigned __int8)sub_250C180(*v5, *a3) )
      return v5;
    if ( !(unsigned __int8)sub_250C180(v5[2], *a3) )
      return v5 + 2;
    if ( !(unsigned __int8)sub_250C180(v5[4], *a3) )
      return v5 + 4;
    if ( !(unsigned __int8)sub_250C180(v5[6], *a3) )
      return v5 + 6;
    v5 += 8;
    if ( v5 == v7 )
    {
      v6 = (a2 - (__int64)v5) >> 4;
      goto LABEL_11;
    }
  }
}
