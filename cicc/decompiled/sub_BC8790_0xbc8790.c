// Function: sub_BC8790
// Address: 0xbc8790
//
char __fastcall sub_BC8790(__int64 a1)
{
  char result; // al
  unsigned __int8 v2; // al
  __int64 v3; // rax

  result = sub_BC8680(a1);
  if ( result )
  {
    v2 = *(_BYTE *)(a1 - 16);
    if ( (v2 & 2) != 0 )
      v3 = *(_QWORD *)(a1 - 32);
    else
      v3 = a1 - 8LL * ((v2 >> 2) & 0xF) - 16;
    return **(_BYTE **)(v3 + 8) == 0;
  }
  return result;
}
