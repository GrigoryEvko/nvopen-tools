// Function: sub_2E88ED0
// Address: 0x2e88ed0
//
char __fastcall sub_2E88ED0(__int64 a1, int a2)
{
  int v2; // eax
  char result; // al
  unsigned __int16 v4; // ax
  unsigned __int16 v5; // ax

  if ( a2 && (v2 = *(_DWORD *)(a1 + 44), (v2 & 4) == 0) && (v2 & 8) != 0 )
  {
    result = sub_2E88A90(a1, 128, a2);
    if ( result )
    {
      v5 = *(_WORD *)(a1 + 68);
      if ( v5 > 0x1Cu )
        return v5 != 32;
      else
        return v5 <= 0x19u;
    }
  }
  else
  {
    result = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 7;
    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & 0x80u) != 0LL )
    {
      v4 = *(_WORD *)(a1 + 68);
      if ( v4 > 0x1Cu )
        return v4 != 32;
      else
        return v4 <= 0x19u;
    }
  }
  return result;
}
