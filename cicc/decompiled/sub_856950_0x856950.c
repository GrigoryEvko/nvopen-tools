// Function: sub_856950
// Address: 0x856950
//
__int64 __fastcall sub_856950(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v6; // r14
  int v7; // r15d
  __int64 result; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // kr00_8
  unsigned __int16 v12; // [rsp+8h] [rbp-48h]
  unsigned int v13; // [rsp+8h] [rbp-48h]
  unsigned int v14; // [rsp+Ch] [rbp-44h]
  unsigned int v15[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v14 = a1;
  v6 = word_4F06418[0];
  v7 = dword_4D03CE4;
  if ( word_4F06418[0] != 10 )
  {
    a6 = (unsigned int)dword_4D03CE0;
    if ( dword_4D03CE0 )
      goto LABEL_5;
    a2 = 14;
    a1 = 7;
    sub_684AA0(7u, 0xEu, &dword_4F063F8);
    v6 = word_4F06418[0];
    while ( (unsigned __int16)(v6 - 9) > 1u )
    {
      sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
      v6 = word_4F06418[0];
LABEL_5:
      ;
    }
  }
  dword_4D03CE4 = 1;
  unk_4D03D20 = 1;
  dword_4D03D1C = 0;
LABEL_7:
  result = (__int64)&dword_4D03D18;
  dword_4D03D18 = 0;
  while ( v6 != 9 )
  {
    result = sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
    if ( (_WORD)result == 68 )
    {
      dword_4D03CE0 = 0;
      unk_4D03CF8 = 1;
      v12 = word_4F063FC[0];
      dword_4D03D18 = 1;
      dword_4D03CF4 = 1;
      sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
      unk_4D03CF8 = 0;
      v6 = word_4F06418[0];
      qword_4F5FCC0 = *(_QWORD *)&dword_4F063F8;
      if ( word_4F06418[0] != 13 && word_4F06418[0] != 10 )
      {
        if ( word_4F06418[0] == 1 )
        {
          v9 = sub_855880();
          v11 = v10;
          a3 = v9;
          switch ( v9 )
          {
            case 0u:
            case 1u:
            case 2u:
              do
                sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
              while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u );
              sub_855540(a1, a2, a3, a4, a5, a6);
              a1 = 0;
              sub_856950(0);
              v6 = word_4F06418[0];
              goto LABEL_7;
            case 3u:
            case 5u:
            case 6u:
              a1 = 0;
              v13 = v9;
              sub_856D20(0);
              a2 = v14;
              if ( !v14 )
                goto LABEL_37;
              if ( v13 == 3 )
              {
                a1 = (unsigned __int64)v15;
                result = (__int64)sub_855790(v15, (unsigned int *)v14);
              }
              else
              {
                a2 = (__int64)v15;
                a1 = v13 == 5;
                result = sub_855EF0(a1, v15, 0, a4, a5, a6);
              }
              a4 = v15[0];
              if ( v15[0] )
                goto LABEL_12;
              v6 = word_4F06418[0];
              if ( word_4F06418[0] != 10 )
              {
                a3 = (unsigned int)dword_4D03CE0;
                if ( !dword_4D03CE0 )
                {
                  a2 = 14;
                  a1 = 7;
                  sub_684AA0(7u, 0xEu, &dword_4F063F8);
                  v6 = word_4F06418[0];
                }
                if ( (unsigned __int16)(v6 - 9) > 1u )
                {
                  do
                  {
                    sub_7B8B50(a1, (unsigned int *)a2, a3, a4, a5, a6);
                    v6 = word_4F06418[0];
                  }
                  while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u );
                }
              }
              break;
            case 4u:
              result = sub_856E70(0);
              a1 = v14;
              if ( !v14 )
                goto LABEL_37;
              goto LABEL_12;
            case 7u:
              sub_855E20(a1, (unsigned int *)a2, v9, a4, a5, a6);
              result = (__int64)dword_4D03C98;
              if ( dword_4D03C98[0] )
              {
                result = (__int64)qword_4F064B0;
                if ( (qword_4F064B0[11] & 1) == 0 && *((_DWORD *)qword_4F064B0 + 20) == dword_4D03CA0 )
                {
                  result = v12;
                  if ( unk_4D03CA4 == v12 )
                    dword_4F5FCB8 = 1;
                }
              }
              goto LABEL_12;
            case 0x16u:
              goto LABEL_16;
            default:
              a3 = v11;
              goto LABEL_7;
          }
        }
        else
        {
LABEL_16:
          if ( dword_4D04964 )
          {
            a2 = (__int64)dword_4F07508;
            a1 = 11;
            sub_684B30(0xBu, dword_4F07508);
            dword_4D03CE0 = 1;
LABEL_37:
            v6 = word_4F06418[0];
          }
        }
      }
      goto LABEL_7;
    }
    v6 = word_4F06418[0];
  }
  dword_4D03CE0 = 1;
LABEL_12:
  dword_4D03CE4 = v7;
  return result;
}
