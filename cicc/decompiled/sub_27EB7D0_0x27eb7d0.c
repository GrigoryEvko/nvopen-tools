// Function: sub_27EB7D0
// Address: 0x27eb7d0
//
__int64 *__fastcall sub_27EB7D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 *result; // rax
  char v16; // r8
  char v17; // r8
  bool v18; // zf

  v5 = (a2 - (__int64)a1) >> 7;
  v6 = a1;
  v7 = (a2 - (__int64)a1) >> 5;
  if ( v5 <= 0 )
  {
LABEL_11:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (__int64 *)a2;
LABEL_19:
        v18 = (unsigned __int8)sub_D48480(a3, *v6, a3, a4) == 0;
        result = v6;
        if ( !v18 )
          return (__int64 *)a2;
        return result;
      }
      v16 = sub_D48480(a3, *v6, a3, a4);
      result = v6;
      if ( !v16 )
        return result;
      v6 += 4;
    }
    v17 = sub_D48480(a3, *v6, a3, a4);
    result = v6;
    if ( !v17 )
      return result;
    v6 += 4;
    goto LABEL_19;
  }
  v8 = &a1[16 * v5];
  while ( 1 )
  {
    if ( !(unsigned __int8)sub_D48480(a3, *v6, a3, a4) )
      return v6;
    if ( !(unsigned __int8)sub_D48480(a3, v6[4], v13, v14) )
      return v6 + 4;
    if ( !(unsigned __int8)sub_D48480(a3, v6[8], v9, v10) )
      return v6 + 8;
    if ( !(unsigned __int8)sub_D48480(a3, v6[12], v11, v12) )
      return v6 + 12;
    v6 += 16;
    if ( v6 == v8 )
    {
      v7 = (a2 - (__int64)v6) >> 5;
      goto LABEL_11;
    }
  }
}
