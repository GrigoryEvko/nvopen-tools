// Function: sub_6421F0
// Address: 0x6421f0
//
void __fastcall sub_6421F0(_BYTE *a1, _QWORD *a2, int a3, int a4, int a5, int a6)
{
  char v6; // al
  char v7; // dl

  if ( ((a1[170] & 0x10) == 0 || unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0)
    && (a1[174] & 4) == 0
    && (*(a1 - 8) & 0x10) == 0 )
  {
    if ( a4 )
    {
      if ( (a1[156] & 4) == 0 )
        sub_6868B0(5, 3500, a2 + 6, "__constant__", *a2);
    }
    else if ( (a5 & a3) != 0 )
    {
      if ( (*((_WORD *)a1 + 78) & 0x101) != 0x101 )
        sub_6868B0(5, 3500, a2 + 6, "__managed__", *a2);
    }
    else if ( a6 )
    {
      if ( (a1[156] & 2) == 0 )
        sub_6868B0(5, 3500, a2 + 6, "__shared__", *a2);
    }
    else if ( a3 )
    {
      v6 = a1[156];
      if ( (v6 & 4) != 0 )
      {
        sub_6870E0(5, 3499, a2 + 6, "__device__", "__constant__", *a2);
      }
      else if ( (*((_WORD *)a1 + 78) & 0x101) == 0x101 )
      {
        sub_6870E0(5, 3499, a2 + 6, "__device__", "__managed__", *a2);
      }
      else if ( (v6 & 2) != 0 )
      {
        sub_6870E0(5, 3499, a2 + 6, "__device__", "__shared__", *a2);
      }
      else if ( (v6 & 1) == 0 )
      {
        sub_6868B0(5, 3500, a2 + 6, "__device__", *a2);
      }
    }
    else if ( !a5 )
    {
      v7 = a1[156];
      if ( (v7 & 4) != 0 )
      {
        sub_6870E0(5, 3499, a2 + 6, "host", "__constant__", *a2);
      }
      else if ( (*((_WORD *)a1 + 78) & 0x101) == 0x101 )
      {
        sub_6870E0(5, 3499, a2 + 6, "host", "__managed__", *a2);
      }
      else if ( (v7 & 2) != 0 )
      {
        sub_6870E0(5, 3499, a2 + 6, "host", "__shared__", *a2);
      }
      else if ( (a1[156] & 1) != 0 )
      {
        sub_6870E0(5, 3499, a2 + 6, "host", "__device__", *a2);
      }
    }
  }
}
