// Function: sub_6F6BD0
// Address: 0x6f6bd0
//
void __fastcall sub_6F6BD0(__int64 *a1, int a2)
{
  unsigned int v3; // esi
  __int64 v4; // rdi

  if ( dword_4F077C4 != 2 || a2 )
  {
    v3 = 0;
LABEL_4:
    sub_6F69D0(a1, v3);
    return;
  }
  if ( unk_4F07778 > 201102 || dword_4F077BC | dword_4F07774 )
  {
    v4 = *a1;
    if ( (*(_BYTE *)(*a1 + 140) & 0xFB) == 8 && (sub_8D4C10(v4, 0) & 2) != 0 && *((_BYTE *)a1 + 16) == 1 )
    {
      v3 = 3;
      if ( (unsigned int)sub_6DEC70(a1[18]) )
        goto LABEL_4;
    }
  }
  sub_6F69D0(a1, 7u);
}
