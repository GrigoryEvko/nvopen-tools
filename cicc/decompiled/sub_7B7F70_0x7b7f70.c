// Function: sub_7B7F70
// Address: 0x7b7f70
//
__int64 __fastcall sub_7B7F70(char *a1)
{
  char v1; // al
  unsigned int v2; // r8d

  v1 = *a1;
  if ( unk_4D043A8 )
  {
    if ( v1 == 117 )
    {
      v1 = a1[1];
      if ( v1 == 56 )
      {
        v1 = a1[2];
        v2 = 2;
        a1 += 2;
      }
      else
      {
        ++a1;
        v2 = 3;
      }
      goto LABEL_6;
    }
    if ( v1 == 85 )
    {
      v1 = a1[1];
      v2 = 4;
      ++a1;
      goto LABEL_6;
    }
  }
  v2 = 1;
  if ( v1 == 76 )
  {
    v1 = a1[1];
    v2 = 5;
    ++a1;
  }
LABEL_6:
  if ( unk_4F07710 && v1 == 82 )
  {
    v1 = a1[1];
    v2 |= 8u;
  }
  if ( v1 == 34 )
    return v2 | 0x10;
  if ( v1 != 39 || (v2 & 8) != 0 )
    return (unsigned int)-1;
  if ( v2 != 2 )
    return v2;
  if ( !unk_4D041F0 )
    return (unsigned int)-1;
  return v2;
}
