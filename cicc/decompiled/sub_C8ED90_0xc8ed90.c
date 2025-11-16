// Function: sub_C8ED90
// Address: 0xc8ed90
//
__int64 __fastcall sub_C8ED90(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  int v4; // edx

  v2 = *a1;
  v3 = -1431655765 * ((a1[1] - *a1) >> 3);
  if ( !v3 )
    return v3;
  v4 = 0;
  while ( a2 < *(_QWORD *)(*(_QWORD *)v2 + 8LL) || a2 > *(_QWORD *)(*(_QWORD *)v2 + 16LL) )
  {
    ++v4;
    v2 += 24;
    if ( v3 == v4 )
      return 0;
  }
  return (unsigned int)(v4 + 1);
}
