// Function: sub_2103340
// Address: 0x2103340
//
__int64 __fastcall sub_2103340(__int64 a1, unsigned int a2)
{
  __int64 v2; // rcx
  __int16 v3; // dx
  unsigned int v4; // esi
  _WORD *v5; // r8
  __int16 *v6; // rcx
  unsigned __int16 v7; // si
  __int16 *v8; // rdx
  __int16 v9; // ax

  v2 = *(_QWORD *)(a1 + 232);
  v3 = a2;
  if ( !v2 )
    BUG();
  v4 = *(_DWORD *)(*(_QWORD *)(v2 + 8) + 24LL * a2 + 16);
  v5 = (_WORD *)(*(_QWORD *)(v2 + 56) + 2LL * (v4 >> 4));
  v6 = v5 + 1;
  v7 = *v5 + v3 * (v4 & 0xF);
LABEL_3:
  v8 = v6;
  if ( !v6 )
    return 0;
  while ( !*(_DWORD *)(*(_QWORD *)(a1 + 384) + 216LL * v7 + 204) )
  {
    v9 = *v8;
    v6 = 0;
    ++v8;
    if ( !v9 )
      goto LABEL_3;
    v7 += v9;
    if ( !v8 )
      return 0;
  }
  return 1;
}
