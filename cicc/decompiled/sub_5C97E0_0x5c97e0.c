// Function: sub_5C97E0
// Address: 0x5c97e0
//
__int64 __fastcall sub_5C97E0(_BYTE *a1)
{
  __int64 result; // rax
  char v2; // dl

  result = 0;
  if ( *a1 == 109 )
  {
    result = (unsigned int)(unk_4D04548 | unk_4D04558);
    if ( unk_4D04548 | unk_4D04558 )
    {
      v2 = a1[1];
      if ( v2 == 120 )
        goto LABEL_8;
      if ( v2 == 99 )
      {
        if ( unk_4F077C4 == 2 )
          return 0;
        goto LABEL_8;
      }
      result = 0;
      if ( v2 == 43 && unk_4F077C4 == 2 )
      {
LABEL_8:
        result = 1;
        if ( a1[2] == 40 )
          return sub_5C9690(qword_4F07788, a1 + 2);
      }
    }
  }
  return result;
}
