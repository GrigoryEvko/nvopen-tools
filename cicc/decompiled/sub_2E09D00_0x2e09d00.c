// Function: sub_2E09D00
// Address: 0x2e09d00
//
__int64 __fastcall sub_2E09D00(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rdx
  unsigned int v4; // edi
  __int64 v5; // rcx
  __int64 v6; // rsi

  v2 = *a1;
  v3 = *((unsigned int *)a1 + 2);
  if ( 3 * v3 )
  {
    v4 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v5 = v3 >> 1;
        v6 = v2 + 8 * ((v3 >> 1) + (v3 & 0xFFFFFFFFFFFFFFFELL));
        if ( (*(_DWORD *)((*(_QWORD *)(v6 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)(v6 + 8) >> 1) & 3) > v4 )
          break;
        v2 = v6 + 24;
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
