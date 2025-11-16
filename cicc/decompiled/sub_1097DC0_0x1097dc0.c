// Function: sub_1097DC0
// Address: 0x1097dc0
//
_BYTE *__fastcall sub_1097DC0(_QWORD *a1)
{
  _BYTE *v1; // r8
  _BYTE *v2; // rdx
  _BYTE *v3; // rcx

  v1 = (_BYTE *)a1[19];
  a1[13] = v1;
  if ( *v1 != 10 && *v1 != 13 )
  {
    v2 = v1;
    v3 = (_BYTE *)(a1[20] + a1[21]);
    while ( v3 != v2 )
    {
      a1[19] = ++v2;
      if ( *v2 == 10 || *v2 == 13 )
        return v1;
    }
  }
  return v1;
}
