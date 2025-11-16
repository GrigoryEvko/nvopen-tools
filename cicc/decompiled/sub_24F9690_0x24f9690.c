// Function: sub_24F9690
// Address: 0x24f9690
//
_QWORD *__fastcall sub_24F9690(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r8
  __int64 i; // rax
  _QWORD *v4; // rcx

  v2 = *(_QWORD **)a1;
  for ( i = *(unsigned int *)(a1 + 8); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v4 = &v2[i >> 1];
      if ( *v4 >= *a2 )
        break;
      v2 = v4 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v2;
    }
  }
  return v2;
}
