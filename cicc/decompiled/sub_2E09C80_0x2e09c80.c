// Function: sub_2E09C80
// Address: 0x2e09c80
//
_QWORD *__fastcall sub_2E09C80(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r9
  __int64 v3; // rdx
  unsigned int v4; // r8d
  __int64 v5; // rcx
  __int64 *v6; // rsi

  v2 = *(_QWORD **)a1;
  v3 = *(unsigned int *)(a1 + 8);
  if ( 3 * v3 )
  {
    v4 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v5 = v3 >> 1;
        v6 = &v2[(v3 >> 1) + (v3 & 0xFFFFFFFFFFFFFFFELL)];
        if ( v4 < (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3) )
          break;
        v2 = v6 + 3;
        v3 = v3 - v5 - 1;
        if ( v3 <= 0 )
          return v2;
      }
      v3 >>= 1;
    }
    while ( v5 > 0 );
  }
  return v2;
}
