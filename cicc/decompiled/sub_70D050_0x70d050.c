// Function: sub_70D050
// Address: 0x70d050
//
unsigned int *__fastcall sub_70D050(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned int *result; // rax

  sub_72D3B0(a1, a2, a3);
  v6 = *(_QWORD *)(a1 + 152);
  v7 = *(_QWORD *)(a1 + 152);
  if ( *(_BYTE *)(v6 + 140) == 12 )
  {
    do
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
  }
  if ( !*(_QWORD *)(*(_QWORD *)(v7 + 168) + 40LL)
    && (*(_BYTE *)(a1 + 89) & 4) != 0
    && (result = *(unsigned int **)(*(_QWORD *)(a1 + 40) + 32LL), (*((_BYTE *)result + 177) & 0x20) != 0)
    || (result = &dword_4D03F94, !dword_4D03F94)
    && (!dword_4F07588
     || dword_4F04C44 != -1
     || (result = (unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64), (*((_BYTE *)result + 6) & 6) != 0)
     || *((_BYTE *)result + 4) == 12)
    && (result = (unsigned int *)sub_8DBE70(v6), (_DWORD)result) )
  {
    *a4 = 1;
  }
  return result;
}
