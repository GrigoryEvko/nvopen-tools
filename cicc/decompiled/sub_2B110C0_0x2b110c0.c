// Function: sub_2B110C0
// Address: 0x2b110c0
//
__int64 *__fastcall sub_2B110C0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // rbx
  _BYTE *v7; // rdi
  _BYTE *v8; // rdi
  _BYTE *v9; // rdi

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
        if ( *(_BYTE *)*v2 != 13 && !sub_B527F0(*v2) )
          return v2;
        return (__int64 *)a2;
      }
      if ( *(_BYTE *)*v2 != 13 && !sub_B527F0(*v2) )
        return v2;
      ++v2;
    }
    if ( *(_BYTE *)*v2 != 13 && !sub_B527F0(*v2) )
      return v2;
    ++v2;
    goto LABEL_25;
  }
  v5 = &a1[4 * v3];
  while ( *(_BYTE *)*v2 == 13 || sub_B527F0(*v2) )
  {
    v7 = (_BYTE *)v2[1];
    if ( *v7 != 13 && !sub_B527F0((__int64)v7) )
      return v2 + 1;
    v8 = (_BYTE *)v2[2];
    if ( *v8 != 13 && !sub_B527F0((__int64)v8) )
    {
      v2 += 2;
      return v2;
    }
    v9 = (_BYTE *)v2[3];
    if ( *v9 != 13 && !sub_B527F0((__int64)v9) )
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
