// Function: sub_1F1B250
// Address: 0x1f1b250
//
unsigned __int64 __fastcall sub_1F1B250(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r14
  __int64 *v4; // rdx
  int *v5; // r10
  __int64 v6; // rdx
  __int64 v7; // rax

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v4 = (__int64 *)sub_1DB3C70((__int64 *)v3, a2 & 0xFFFFFFFFFFFFFFF8LL | 6);
  if ( v4 == (__int64 *)(*(_QWORD *)v3 + 24LL * *(unsigned int *)(v3 + 8)) )
    return a2 & 0xFFFFFFFFFFFFFFF8LL | 6;
  if ( (*(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v4 >> 1) & 3) > (*(_DWORD *)(v2 + 24) | 3u) )
    return a2 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v5 = (int *)v4[2];
  if ( !v5 )
    return a2 & 0xFFFFFFFFFFFFFFF8LL | 6;
  if ( !v2 || (v6 = *(_QWORD *)(v2 + 16)) == 0 )
    BUG();
  v7 = *(_QWORD *)(v2 + 16);
  if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 46) & 8) != 0 )
  {
    do
      v7 = *(_QWORD *)(v7 + 8);
    while ( (*(_BYTE *)(v7 + 46) & 8) != 0 );
  }
  return *(_QWORD *)(sub_1F1AD70(
                       (_QWORD *)a1,
                       *(_DWORD *)(a1 + 80),
                       v5,
                       a2 & 0xFFFFFFFFFFFFFFF8LL | 6,
                       *(_QWORD *)(v6 + 24),
                       *(unsigned __int64 **)(v7 + 8))
                   + 8);
}
