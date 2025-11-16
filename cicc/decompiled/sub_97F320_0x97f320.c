// Function: sub_97F320
// Address: 0x97f320
//
__int64 __fastcall sub_97F320(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r8d
  __int16 v5; // ax
  int v6; // eax

  v1 = *(_QWORD *)(a1 + 80);
  v2 = sub_B43CA0(a1);
  v5 = (*(_WORD *)(a1 + 2) >> 2) & 0x3FF;
  if ( !v5 )
    return 1;
  if ( (unsigned __int16)(v5 - 66) > 2u )
    return 0;
  v6 = *(_DWORD *)(v2 + 276);
  if ( v6 == 27 || v6 == 5 )
    return 0;
  return sub_97E0F0(*(_DWORD *)(v1 + 12), *(_QWORD *)(v1 + 16), v2, v3, v4);
}
