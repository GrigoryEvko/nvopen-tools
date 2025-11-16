// Function: sub_750540
// Address: 0x750540
//
_QWORD *__fastcall sub_750540(__int64 a1, __int64 *a2)
{
  unsigned int v2; // edx
  int v3; // eax
  unsigned int v4; // edx
  _QWORD *v5; // r8
  int j; // eax
  unsigned int i; // eax

  v2 = *(_DWORD *)(a1 + 60);
  v3 = *((_DWORD *)a2 + 4);
  if ( v2 )
  {
    if ( v3 || (--v2, v2) )
    {
      for ( i = 0; i < v2; ++i )
        a2 = (__int64 *)*a2;
    }
  }
  else if ( !v3 )
  {
    do
      a2 = (__int64 *)*a2;
    while ( !*((_DWORD *)a2 + 4) );
  }
  v4 = *(_DWORD *)(a1 + 56);
  v5 = (_QWORD *)a2[1];
  if ( v4 > 1 )
  {
    for ( j = 1; j != v4; ++j )
      v5 = (_QWORD *)*v5;
  }
  return v5;
}
