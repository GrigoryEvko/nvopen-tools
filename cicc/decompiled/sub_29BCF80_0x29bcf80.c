// Function: sub_29BCF80
// Address: 0x29bcf80
//
char __fastcall sub_29BCF80(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  int v5; // r13d
  int v6; // r13d
  unsigned int v7; // eax

  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((*a2 >> 2) & 1) == ((*a1 >> 2) & 1) )
    return v2 == v3;
  if ( (unsigned __int8)(*(_BYTE *)v2 - 82) > 1u || (unsigned __int8)(*(_BYTE *)v3 - 82) > 1u )
    return 0;
  v5 = *(_WORD *)(v2 + 2) & 0x3F;
  if ( v5 != (unsigned int)sub_B52870(*(_WORD *)(v3 + 2) & 0x3F)
    || *(_QWORD *)(v2 - 64) != *(_QWORD *)(v3 - 64)
    || *(_QWORD *)(v2 - 32) != *(_QWORD *)(v3 - 32) )
  {
    v6 = *(_WORD *)(v2 + 2) & 0x3F;
    v7 = sub_B52870(*(_WORD *)(v3 + 2) & 0x3F);
    if ( v6 == (unsigned int)sub_B52F50(v7) && *(_QWORD *)(v2 - 64) == *(_QWORD *)(v3 - 32) )
      return *(_QWORD *)(v2 - 32) == *(_QWORD *)(v3 - 64);
    return 0;
  }
  return 1;
}
