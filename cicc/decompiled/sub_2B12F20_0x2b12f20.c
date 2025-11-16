// Function: sub_2B12F20
// Address: 0x2b12f20
//
unsigned __int8 **__fastcall sub_2B12F20(unsigned __int8 **a1, __int64 a2, __int64 a3)
{
  unsigned __int8 **result; // rax
  __int64 v6; // r8
  __int64 v7; // rdx
  unsigned __int8 **v8; // r8
  unsigned __int8 *v9; // rcx
  unsigned __int8 *v10; // rcx
  unsigned __int8 *v11; // rcx
  unsigned __int8 v12; // dl
  unsigned __int8 v13; // dl
  unsigned __int8 v14; // dl
  unsigned __int8 v15; // dl

  result = a1;
  v6 = (a2 - (__int64)a1) >> 7;
  v7 = (a2 - (__int64)a1) >> 5;
  if ( v6 <= 0 )
  {
LABEL_21:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (unsigned __int8 **)a2;
        goto LABEL_33;
      }
      v13 = **result;
      if ( v13 != 84 && v13 > 0x1Cu && *((_QWORD *)*result + 5) == *(_QWORD *)(a3 + 40) )
        return result;
      result += 4;
    }
    v14 = **result;
    if ( v14 != 84 && v14 > 0x1Cu && *((_QWORD *)*result + 5) == *(_QWORD *)(a3 + 40) )
      return result;
    result += 4;
LABEL_33:
    v15 = **result;
    if ( v15 > 0x1Cu && v15 != 84 )
    {
      if ( *((_QWORD *)*result + 5) != *(_QWORD *)(a3 + 40) )
        return (unsigned __int8 **)a2;
      return result;
    }
    return (unsigned __int8 **)a2;
  }
  v8 = &a1[16 * v6];
  while ( 1 )
  {
    v12 = **result;
    if ( v12 > 0x1Cu && v12 != 84 && *((_QWORD *)*result + 5) == *(_QWORD *)(a3 + 40) )
      return result;
    v9 = result[4];
    if ( *v9 != 84 && *v9 > 0x1Cu && *((_QWORD *)v9 + 5) == *(_QWORD *)(a3 + 40) )
    {
      result += 4;
      return result;
    }
    v10 = result[8];
    if ( *v10 > 0x1Cu && *v10 != 84 && *((_QWORD *)v10 + 5) == *(_QWORD *)(a3 + 40) )
    {
      result += 8;
      return result;
    }
    v11 = result[12];
    if ( *v11 > 0x1Cu && *v11 != 84 && *((_QWORD *)v11 + 5) == *(_QWORD *)(a3 + 40) )
    {
      result += 12;
      return result;
    }
    result += 16;
    if ( v8 == result )
    {
      v7 = (a2 - (__int64)result) >> 5;
      goto LABEL_21;
    }
  }
}
