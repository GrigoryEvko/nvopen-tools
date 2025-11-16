// Function: sub_2EBF080
// Address: 0x2ebf080
//
_BOOL8 __fastcall sub_2EBF080(__int64 a1, unsigned int a2, int a3)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx

  v3 = a2;
  if ( (int)v3 < 0 )
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16 * (v3 & 0x7FFFFFFF) + 8);
  else
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8 * v3);
  while ( 1 )
  {
    if ( !v5 )
      return a3 != -1;
    if ( (*(_BYTE *)(v5 + 3) & 0x10) == 0 && (*(_BYTE *)(v5 + 4) & 8) == 0 )
      break;
    v5 = *(_QWORD *)(v5 + 32);
  }
  v7 = (unsigned int)(a3 + 1);
  if ( !(_DWORD)v7 )
    return 0;
  while ( 1 )
  {
    LODWORD(v7) = v7 - (unsigned __int8)sub_2EBDFB0(*(_QWORD *)(v5 + 16), v7);
    do
      v5 = *(_QWORD *)(v5 + 32);
    while ( v5 && ((*(_BYTE *)(v5 + 3) & 0x10) != 0 || (*(_BYTE *)(v5 + 4) & 8) != 0 || v8 == *(_QWORD *)(v5 + 16)) );
    if ( !(_DWORD)v7 )
      break;
    if ( !v5 )
      return 1;
  }
  return 0;
}
