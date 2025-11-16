// Function: sub_2B08630
// Address: 0x2b08630
//
char __fastcall sub_2B08630(__int64 a1)
{
  __int64 v1; // rbx
  char result; // al

  v1 = a1;
  if ( (_BYTE)qword_5010508 && *(_BYTE *)(a1 + 8) == 17 )
    v1 = **(_QWORD **)(a1 + 16);
  result = sub_BCBCB0(v1);
  if ( result )
    return (*(_BYTE *)(v1 + 8) & 0xFD) != 4;
  return result;
}
