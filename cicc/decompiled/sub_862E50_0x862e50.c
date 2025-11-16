// Function: sub_862E50
// Address: 0x862e50
//
__int64 __fastcall sub_862E50(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx

  result = 0;
  if ( a1 )
  {
    if ( (*(_BYTE *)(a1 + 207) & 0x40) != 0
      || (*(_BYTE *)(a1 + 89) & 4) != 0
      && ((v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), (*(_BYTE *)(v2 + 141) & 0x20) != 0)
       || (v3 = *(_QWORD *)(v2 + 168), (*(_BYTE *)(v3 + 109) & 0x20) != 0)
       && ((*(_BYTE *)(v3 + 109) & 0x40) != 0
        || (*(_BYTE *)(a1 + 206) & 2) != 0
        && (*(_BYTE *)(v3 + 110) & 8) != 0
        && (*(_BYTE *)(*(_QWORD *)(v3 + 240) + 170LL) & 2) != 0)) )
    {
      unk_4F04C20 = 1;
      return 1;
    }
  }
  return result;
}
