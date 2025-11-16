// Function: sub_DF72F0
// Address: 0xdf72f0
//
__int64 __fastcall sub_DF72F0(__int64 a1, int *a2, int *a3)
{
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 18 )
    return 0;
  else
    return sub_B4F0B0(
             *(int **)(a1 + 72),
             *(unsigned int *)(a1 + 80),
             *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL),
             a2,
             a3);
}
