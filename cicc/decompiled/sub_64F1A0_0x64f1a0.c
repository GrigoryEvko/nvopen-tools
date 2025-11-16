// Function: sub_64F1A0
// Address: 0x64f1a0
//
__int64 __fastcall sub_64F1A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int16 v5; // ax
  unsigned int v6; // eax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rbx
  int *v11; // rax
  int v12; // edx
  int *v13; // rax
  bool v14; // [rsp+7h] [rbp-59h]
  int v16; // [rsp+1Ch] [rbp-44h] BYREF
  int v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+24h] [rbp-3Ch] BYREF
  _QWORD v19[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( !a3 || *(_BYTE *)(a3 + 40) )
  {
    if ( a2 && *(_BYTE *)(a2 + 80) == 11 )
    {
      v14 = 1;
      goto LABEL_34;
    }
    v8 = a1 + 48;
    v9 = 3775;
    return sub_684AA0(8, v9, v8);
  }
  if ( a2 )
    v14 = *(_BYTE *)(a2 + 80) == 11;
  else
    v14 = 0;
  if ( !unk_4F04C50 || (v3 = *(_QWORD *)(unk_4F04C50 + 32LL)) == 0 || (*(_BYTE *)(v3 + 198) & 0x10) == 0 )
  {
    v8 = a1 + 48;
    v9 = 3776;
    return sub_684AA0(8, v9, v8);
  }
  if ( !a2 || *(_BYTE *)(a2 + 80) != 11 )
    goto LABEL_10;
LABEL_34:
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 88) + 198LL) & 0x30) != 0x10 )
  {
    v8 = a1 + 48;
    v9 = 3777;
    return sub_684AA0(8, v9, v8);
  }
LABEL_10:
  sub_7C9660(a1);
  v16 = -1;
  v17 = -1;
  v18 = -1;
  v19[0] = *(_QWORD *)&dword_4F063F8;
  v5 = word_4F06418[0];
  if ( word_4F06418[0] != 9 )
  {
    while ( 1 )
    {
      if ( v5 != 1 )
      {
LABEL_38:
        v6 = 3780;
        goto LABEL_37;
      }
      if ( !strcmp(*(const char **)(qword_4D04A00 + 8), "preserve_n_data") )
      {
        v6 = sub_642070(&v16, (__int64)v19, v4);
        if ( v6 )
          goto LABEL_19;
      }
      else if ( !strcmp(*(const char **)(qword_4D04A00 + 8), "preserve_n_control") )
      {
        v6 = sub_642070(&v17, (__int64)v19, v4);
        if ( v6 )
          goto LABEL_19;
      }
      else
      {
        if ( strcmp(*(const char **)(qword_4D04A00 + 8), "preserve_n_after") )
          goto LABEL_38;
        v6 = sub_642070(&v18, (__int64)v19, v4);
        if ( v6 )
        {
LABEL_19:
          if ( v6 != 125 && v6 != 18 )
            goto LABEL_37;
          return sub_7C96B0(1);
        }
      }
      v19[0] = *(_QWORD *)&dword_4F063F8;
      v5 = word_4F06418[0];
      if ( word_4F06418[0] == 9 )
      {
        if ( v14 )
        {
          if ( a2 )
          {
            v10 = *(_QWORD *)(a2 + 88);
            if ( v10 )
            {
              v11 = (int *)sub_725FB0();
              v12 = v16;
              *(_QWORD *)(v10 + 336) = v11;
              *v11 = v12;
              *(_DWORD *)(*(_QWORD *)(v10 + 336) + 4LL) = v17;
              *(_DWORD *)(*(_QWORD *)(v10 + 336) + 8LL) = v18;
            }
          }
        }
        else
        {
          v13 = (int *)sub_725FB0();
          *v13 = v16;
          v13[1] = v17;
          v13[2] = v18;
          *(_QWORD *)(a3 + 72) = v13;
        }
        return sub_7C96B0(1);
      }
    }
  }
  v6 = 3778;
LABEL_37:
  sub_684AA0(8, v6, v19);
  return sub_7C96B0(1);
}
