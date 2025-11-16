// Function: sub_1DB3C70
// Address: 0x1db3c70
//
__int64 __fastcall sub_1DB3C70(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // rcx
  unsigned int v5; // esi
  unsigned __int64 v7; // rcx

  v2 = *((unsigned int *)a1 + 2);
  v3 = *a1;
  v4 = 24 * v2;
  if ( (_DWORD)v2
    && (v5 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3,
        v5 < (*(_DWORD *)((*(_QWORD *)(v3 + v4 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)(v3 + v4 - 16) >> 1) & 3)) )
  {
    do
    {
      v7 = 8 * ((v2 >> 1) + (v2 & 0xFFFFFFFFFFFFFFFELL));
      if ( v5 >= (*(_DWORD *)((*(_QWORD *)(v3 + v7 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(v3 + v7 + 8) >> 1) & 3) )
      {
        v3 += v7 + 24;
        v2 = v2 - 1 - (v2 >> 1);
      }
      else
      {
        v2 >>= 1;
      }
    }
    while ( v2 );
  }
  else
  {
    v3 += v4;
  }
  return v3;
}
