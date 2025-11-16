// Function: sub_2FEDBC0
// Address: 0x2fedbc0
//
__int64 __fastcall sub_2FEDBC0(void *a1, __int64 a2)
{
  __int64 v2; // r8
  bool v3; // zf
  bool v5; // zf

  if ( &unk_5022FAC == a1 )
  {
    v3 = (_BYTE)qword_5029FE8 == 0;
    goto LABEL_19;
  }
  if ( &unk_503A8EC == a1 )
  {
    v3 = byte_5029F08 == 0;
    goto LABEL_19;
  }
  if ( &unk_5026411 == a1 )
  {
    v3 = (_BYTE)qword_5029E28 == 0;
    goto LABEL_19;
  }
  if ( &unk_5026410 == a1 )
  {
    v3 = (_BYTE)qword_5029D48 == 0;
    goto LABEL_19;
  }
  if ( &unk_503BDC4 == a1 )
  {
    v3 = (_BYTE)qword_5029C68 == 0;
    goto LABEL_19;
  }
  if ( &unk_502624C == a1 )
  {
    v3 = byte_5029AA8 == 0;
    goto LABEL_19;
  }
  if ( &unk_501CF4C == a1 )
  {
    v3 = byte_50299C8 == 0;
LABEL_19:
    v2 = 0;
    if ( v3 )
      return a2;
    return v2;
  }
  if ( &unk_501CF6C == a1 )
  {
    v5 = byte_50298E8 == 0;
  }
  else if ( &unk_50201E8 == a1 )
  {
    v5 = byte_5029808 == 0;
  }
  else
  {
    if ( &unk_501F54C != a1 )
    {
      if ( &unk_50201E9 == a1 )
      {
        v3 = byte_5029568 == 0;
      }
      else if ( &unk_5021D2C == a1 )
      {
        v3 = byte_5029488 == 0;
      }
      else
      {
        if ( &unk_5021D24 != a1 )
        {
          v2 = a2;
          if ( &unk_501F390 == a1 && (_BYTE)qword_5029028 )
            return 0;
          return v2;
        }
        v3 = byte_50293A8 == 0;
      }
      goto LABEL_19;
    }
    v5 = byte_5029728 == 0;
  }
  v2 = 0;
  if ( v5 )
    return a2;
  return v2;
}
