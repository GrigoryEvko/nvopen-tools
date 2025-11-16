// Function: sub_5C6B80
// Address: 0x5c6b80
//
_BYTE *__fastcall sub_5C6B80(__int64 a1, _BYTE *a2, char a3)
{
  char v4; // dl
  char v5; // al
  char *v6; // rdx

  if ( a3 == 7 )
  {
    v4 = a2[156] | 1;
    a2[156] = v4;
    if ( ((v4 & 2) != 0) + ((v4 & 4) != 0) == 2 )
      sub_6851C0(3481, a1 + 56);
    if ( (char)a2[169] < 0 )
      sub_6851C0(3482, a1 + 56);
    if ( (a2[89] & 4) != 0 )
      sub_6851C0(3485, a1 + 56);
    if ( (a2[172] & 8) != 0 && (*((_WORD *)a2 + 78) & 0x102) != 0 )
    {
      v5 = a2[156];
      v6 = "__constant__";
      if ( (v5 & 4) == 0 )
      {
        v6 = "__managed__";
        if ( (a2[157] & 1) == 0 )
        {
          v6 = "__shared__";
          if ( (v5 & 2) == 0 )
          {
            v6 = (char *)byte_3F871B3;
            if ( (v5 & 1) != 0 )
              v6 = "__device__";
          }
        }
      }
      sub_6851A0(3577, a1 + 56, v6);
    }
  }
  else if ( a3 == 11 )
  {
    a2[197] |= 0x80u;
    if ( (a2[89] & 4) != 0 && (a2[198] & 0x20) != 0 )
      sub_6851C0(3688, a1 + 56);
    if ( (_BYTE *)unk_4F07290 == a2 && (a2[198] & 0x10) != 0 )
      sub_684AA0(7, 3538, a1 + 56);
  }
  return a2;
}
