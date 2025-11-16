// Function: sub_97F390
// Address: 0x97f390
//
__int64 __fastcall sub_97F390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int16 v5; // ax
  int v6; // eax

  v5 = (*(_WORD *)(a1 + 2) >> 4) & 0x3FF;
  if ( !v5 )
    return 1;
  if ( (unsigned __int16)(v5 - 66) > 2u )
    return 0;
  v6 = *(_DWORD *)(*(_QWORD *)(a1 + 40) + 276LL);
  if ( v6 == 27 || v6 == 5 )
    return 0;
  return sub_97E0F0(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 12LL), *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL), a3, a4, a5);
}
