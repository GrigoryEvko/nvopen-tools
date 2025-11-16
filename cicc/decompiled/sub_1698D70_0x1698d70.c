// Function: sub_1698D70
// Address: 0x1698d70
//
__int64 __fastcall sub_1698D70(__int64 a1, int a2)
{
  _WORD *v2; // rax
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  __int64 v5; // rax

  if ( (a2 & 0xFFFFFFFB) != 0 )
  {
    if ( a2 == 1 )
    {
      if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
      {
LABEL_5:
        *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
        v2 = *(_WORD **)a1;
        *(_WORD *)(a1 + 16) = **(_WORD **)a1;
        v3 = *((_DWORD *)v2 + 1);
        v4 = sub_1698310(a1);
        v5 = sub_1698470(a1);
        sub_16AEAC0(v5, v4, v3);
        return 16;
      }
    }
    else if ( a2 != 2 || (*(_BYTE *)(a1 + 18) & 8) == 0 )
    {
      goto LABEL_5;
    }
  }
  *(_BYTE *)(a1 + 18) &= 0xF8u;
  return 20;
}
