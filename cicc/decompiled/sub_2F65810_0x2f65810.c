// Function: sub_2F65810
// Address: 0x2f65810
//
__int64 __fastcall sub_2F65810(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r8
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 *v8; // rcx

  v2 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v2 )
    return *(_QWORD *)(v2 + 24);
  v4 = *(unsigned int *)(a1 + 304);
  v5 = *(_QWORD **)(a1 + 296);
  if ( *(_DWORD *)(a1 + 304) )
  {
    v6 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
    do
    {
      while ( 1 )
      {
        v7 = v4 >> 1;
        v8 = &v5[2 * (v4 >> 1)];
        if ( v6 < (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) )
          break;
        v5 = v8 + 2;
        v4 = v4 - v7 - 1;
        if ( v4 <= 0 )
          return *(v5 - 1);
      }
      v4 >>= 1;
    }
    while ( v7 > 0 );
  }
  return *(v5 - 1);
}
