// Function: sub_1438600
// Address: 0x1438600
//
__int64 __fastcall sub_1438600(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v4; // rsi
  unsigned int v5; // eax

  v2 = 7;
  if ( unk_4F99CA8 )
  {
    v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)(v4 + 16) == 78 && !*(_BYTE *)(*(_QWORD *)(v4 - 24) + 16LL) )
    {
      v5 = sub_1438F00();
      if ( v5 <= 0xB )
        return ((1LL << v5) & 0xEE3) == 0 ? 7 : 4;
    }
  }
  return v2;
}
