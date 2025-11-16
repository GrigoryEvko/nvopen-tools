// Function: sub_7BDFF0
// Address: 0x7bdff0
//
__int64 __fastcall sub_7BDFF0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // r12
  int v8; // eax
  __int64 v9; // r14
  unsigned __int16 v10; // r15
  __int64 v11; // rcx
  __int64 result; // rax
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v6 = word_4F06418[0];
  v13 = dword_4F063F8;
  if ( !*(_BYTE *)(a1 + word_4F06418[0]) )
  {
    v7 = a1;
    v8 = word_4F06418[0];
    v9 = 0;
    v10 = 0;
    do
    {
      v11 = v8 & 0xFFFFFFFD;
      if ( (v8 & 0xFFFD) != 0x19 && (_WORD)v8 != 73 )
      {
        if ( (_WORD)v8 != 43 )
          goto LABEL_5;
        if ( v10 == 1 )
        {
          if ( dword_4F077C4 != 2
            || !*(_BYTE *)(v7 + 44) && !*(_BYTE *)(v7 + 67)
            || unk_4D03D20
            || !v9
            || (a1 = v9, !(unsigned int)sub_7AD1A0(v9)) )
          {
LABEL_5:
            v10 = word_4F06418[0];
            if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u )
              break;
            goto LABEL_6;
          }
        }
        else if ( v10 != 160 )
        {
          goto LABEL_5;
        }
      }
      a1 = (_DWORD)a2 == 0;
      sub_7BDC20(a1, (__int64)a2, v6, v11, a5, a6);
      v10 = word_4F06418[0];
      if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u )
        break;
LABEL_6:
      v9 = qword_4D04A00;
      sub_7B8B50(a1, a2, v6, v11, a5, a6);
      v8 = word_4F06418[0];
    }
    while ( !*(_BYTE *)(v7 + word_4F06418[0]) );
  }
  result = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  if ( !(_DWORD)a2 )
  {
    result = dword_4F063F8 - v13;
    if ( (unsigned int)result > 2 )
    {
      result = sub_729AE0(dword_4F063F8);
      if ( !(_DWORD)result )
        return sub_684B30(0xCu, dword_4F07508);
    }
  }
  return result;
}
