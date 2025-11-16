// Function: sub_5D7720
// Address: 0x5d7720
//
int __fastcall sub_5D7720(__int64 a1)
{
  __int64 v1; // rax
  char v2; // al
  int result; // eax
  _BYTE *v4; // rbx
  _BYTE *v5; // r12
  int v6; // edi
  int v7; // r13d

  if ( (*(_BYTE *)(a1 + 202) & 0x40) != 0 )
    return sub_5D34A0();
  if ( (*(_BYTE *)(a1 + 199) & 1) != 0 && sub_5D76E0() && (v4 = *(_BYTE **)(a1 + 136)) != 0 )
  {
    v5 = v4 + 1;
    result = strlen(*(const char **)(a1 + 136));
    v6 = (char)*v4;
    v7 = result;
    if ( *v4 )
    {
      do
      {
        ++v5;
        result = putc(v6, stream);
        v6 = (char)*(v5 - 1);
      }
      while ( *(v5 - 1) );
    }
    dword_4CF7F40 += v7;
  }
  else if ( (*(_BYTE *)(a1 + 197) & 0x60) != 0
         && (v1 = *(_QWORD *)(a1 + 128)) != 0
         && (*(_BYTE *)(v1 + 198) & 0x20) != 0
         && sub_5D76E0() )
  {
    return sub_5D62B0(*(_QWORD *)(a1 + 128));
  }
  else
  {
    v2 = *(_BYTE *)(a1 + 198);
    if ( (v2 & 0x20) != 0 )
    {
      return sub_5D62B0(a1);
    }
    else if ( (v2 & 0x10) != 0 && sub_5D7700() )
    {
      return sub_5D5580(a1, 0);
    }
    else
    {
      return sub_5D5A80(a1, 0);
    }
  }
  return result;
}
