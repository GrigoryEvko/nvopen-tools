// Function: sub_827180
// Address: 0x827180
//
_BOOL8 __fastcall sub_827180(int a1, __int64 *a2)
{
  __int64 v2; // r12
  int v3; // r8d
  _BOOL8 result; // rax
  char v5; // al
  __int64 v6; // rdx

  if ( dword_4F04C44 != -1
    || (v6 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v6 + 6) & 6) != 0)
    || (result = 0, *(_BYTE *)(v6 + 4) == 12) )
  {
    v2 = *a2;
    if ( a1 && (unsigned int)sub_8D2EF0(*a2) )
      v2 = sub_8D46C0(v2);
    v3 = sub_8DD3B0(v2);
    result = 1;
    if ( !v3 )
    {
      while ( 1 )
      {
        v5 = *(_BYTE *)(v2 + 140);
        if ( v5 != 12 )
          break;
        v2 = *(_QWORD *)(v2 + 160);
      }
      return v5 == 0;
    }
  }
  return result;
}
