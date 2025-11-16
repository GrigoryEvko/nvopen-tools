// Function: sub_2B17A10
// Address: 0x2b17a10
//
__int64 *__fastcall sub_2B17A10(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // rbx
  _BYTE *v6; // rdi
  _BYTE *v7; // rdi
  _BYTE *v8; // rdi

  v2 = a1;
  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  if ( v3 <= 0 )
  {
LABEL_17:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return (__int64 *)a2;
LABEL_25:
        if ( *(_BYTE *)*v2 == 44 && sub_2B17690(*v2) )
          return v2;
        return (__int64 *)a2;
      }
      if ( *(_BYTE *)*v2 == 44 && sub_2B17690(*v2) )
        return v2;
      ++v2;
    }
    if ( *(_BYTE *)*v2 == 44 && sub_2B17690(*v2) )
      return v2;
    ++v2;
    goto LABEL_25;
  }
  v5 = &a1[4 * v3];
  while ( *(_BYTE *)*v2 != 44 || !sub_2B17690(*v2) )
  {
    v6 = (_BYTE *)v2[1];
    if ( *v6 == 44 && sub_2B17690((__int64)v6) )
      return v2 + 1;
    v7 = (_BYTE *)v2[2];
    if ( *v7 == 44 && sub_2B17690((__int64)v7) )
      return v2 + 2;
    v8 = (_BYTE *)v2[3];
    if ( *v8 == 44 && sub_2B17690((__int64)v8) )
    {
      v2 += 3;
      return v2;
    }
    v2 += 4;
    if ( v5 == v2 )
    {
      v4 = (a2 - (__int64)v2) >> 3;
      goto LABEL_17;
    }
  }
  return v2;
}
