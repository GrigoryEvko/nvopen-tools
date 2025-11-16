// Function: sub_8C32F0
// Address: 0x8c32f0
//
__int64 __fastcall sub_8C32F0(_QWORD *a1)
{
  __int64 i; // rbx
  __int64 result; // rax
  _QWORD *k; // rbx
  __int64 m; // rbx
  __int64 v6; // rdi
  __int64 j; // rbx
  __int64 v8; // rdi

  for ( i = a1[21]; i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_8C32F0(*(_QWORD *)(i + 128));
  }
  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    for ( j = a1[13]; j; j = *(_QWORD *)(j + 112) )
    {
      result = (unsigned int)*(unsigned __int8 *)(j + 140) - 9;
      if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u )
      {
        result = *(_QWORD *)(j + 168);
        v8 = *(_QWORD *)(result + 152);
        if ( v8 )
          result = sub_8C32F0(v8);
      }
    }
  }
  for ( k = (_QWORD *)a1[20]; k; k = (_QWORD *)*k )
    result = sub_8C32F0(k);
  for ( m = a1[14]; m; m = *(_QWORD *)(m + 112) )
  {
    while ( 1 )
    {
      v6 = m;
      if ( (*(_BYTE *)(m - 8) & 2) != 0 )
      {
        v6 = *(_QWORD *)(m - 24);
        if ( (*(_BYTE *)(v6 - 8) & 2) != 0 )
          v6 = *(_QWORD *)(v6 - 24);
      }
      if ( !*(_BYTE *)(v6 + 136) || *(_BYTE *)(v6 + 177) == 2 )
        break;
      m = *(_QWORD *)(m + 112);
      if ( !m )
        return result;
    }
    result = sub_7604D0(v6, 7u);
  }
  return result;
}
