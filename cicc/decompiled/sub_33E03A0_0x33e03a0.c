// Function: sub_33E03A0
// Address: 0x33e03a0
//
char __fastcall sub_33E03A0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  int v4; // eax
  char result; // al

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 != 187 )
    return v4 == 188 && !a4 && sub_33CF530(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  result = 1;
  if ( (*(_BYTE *)(a2 + 28) & 8) == 0 )
    return sub_33E0180(
             a1,
             **(_QWORD **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  return result;
}
