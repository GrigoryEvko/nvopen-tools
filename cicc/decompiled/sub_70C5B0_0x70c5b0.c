// Function: sub_70C5B0
// Address: 0x70c5b0
//
__int64 __fastcall sub_70C5B0(__int64 a1, unsigned int *a2)
{
  unsigned int v3; // eax
  __int64 v4; // rax
  int v5; // eax
  unsigned int v6; // eax

  if ( dword_4F07890 && (unsigned __int8)(a1 - 5) <= 1u )
    goto LABEL_15;
  if ( (unsigned __int8)a1 <= 2u || (unsigned __int8)(a1 - 9) <= 2u )
    return *a2 >> 31;
  if ( (unsigned __int8)(a1 - 3) <= 1u )
    goto LABEL_15;
  if ( (unsigned __int8)(a1 - 5) <= 1u )
  {
    if ( !dword_4F07890 )
    {
      if ( (_BYTE)a1 != 6 )
      {
        v5 = unk_4F06930;
        if ( unk_4F06930 == 64 )
          goto LABEL_26;
        goto LABEL_22;
      }
      v5 = unk_4F06930;
      if ( unk_4F06930 != 106 )
      {
        if ( unk_4F06930 == 64 )
          goto LABEL_26;
LABEL_22:
        if ( v5 == 113 )
          goto LABEL_12;
        goto LABEL_23;
      }
    }
LABEL_15:
    v4 = HIDWORD(*(_QWORD *)a2);
    if ( !unk_4F07580 )
      LODWORD(v4) = *(_QWORD *)a2;
    return (unsigned int)v4 >> 31;
  }
  if ( (_BYTE)a1 != 14 && (unsigned __int8)a1 > 8u )
  {
    if ( qword_4D040A0[(unsigned __int8)a1] != 8 )
    {
LABEL_11:
      if ( (_BYTE)a1 == 13 )
      {
LABEL_12:
        v3 = *a2;
        if ( unk_4F07580 )
          v3 = a2[3];
        return v3 >> 31;
      }
LABEL_23:
      sub_721090(a1);
    }
    goto LABEL_15;
  }
  if ( (_BYTE)a1 != 7 )
  {
    if ( (_BYTE)a1 == 8 )
    {
      if ( unk_4F06918 == 113 )
        goto LABEL_12;
      goto LABEL_23;
    }
    goto LABEL_11;
  }
  if ( unk_4F06924 != 64 )
    goto LABEL_23;
LABEL_26:
  v6 = a2[2];
  if ( !unk_4F07580 )
    v6 = *(_QWORD *)a2;
  return (v6 >> 15) & 1;
}
