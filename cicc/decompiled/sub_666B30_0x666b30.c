// Function: sub_666B30
// Address: 0x666b30
//
__int64 __fastcall sub_666B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v6; // [rsp-10h] [rbp-30h]

  if ( unk_4D03C90 )
    sub_858B20();
  if ( unk_4F076D0 )
    sub_858BD0();
  unk_4F04D84 = 1;
  sub_7B8B50(a1, a2, a3, a4);
  unk_4F04D84 = 0;
  if ( unk_4D03C84 )
    sub_852DF0();
  if ( word_4F06418[0] != 9 )
  {
    while ( 1 )
    {
      if ( dword_4F077C4 == 2 )
      {
        if ( unk_4F07778 > 201102 || dword_4F07774 )
        {
LABEL_11:
          ((void (*)(void))sub_857CE0)();
          goto LABEL_12;
        }
      }
      else if ( unk_4F07778 > 199900 )
      {
        goto LABEL_11;
      }
      if ( unk_4D04778 )
        goto LABEL_11;
LABEL_12:
      a2 = 0;
      a1 = 1;
      sub_660E20(1, 0, 1, 0, 0, 0, 0);
      v4 = v6;
      if ( word_4F06418[0] == 9 )
        goto LABEL_21;
    }
  }
  if ( dword_4F077C4 == 2 )
    goto LABEL_26;
  if ( !dword_4D04964 || unk_4D03C90 )
  {
    if ( unk_4F07778 <= 199900 )
      return sub_854C70(a1, a2, v4);
    goto LABEL_23;
  }
  a2 = 96;
  a1 = byte_4F07472[0];
  sub_684AC0(byte_4F07472[0], 96);
LABEL_21:
  if ( dword_4F077C4 == 2 )
  {
LABEL_26:
    if ( unk_4F07778 > 201102 || dword_4F07774 )
      goto LABEL_23;
  }
  else if ( unk_4F07778 > 199900 )
  {
LABEL_23:
    sub_857CE0(a1, a2, v4);
  }
  return sub_854C70(a1, a2, v4);
}
