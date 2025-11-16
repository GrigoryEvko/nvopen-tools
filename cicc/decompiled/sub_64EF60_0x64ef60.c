// Function: sub_64EF60
// Address: 0x64ef60
//
char *__fastcall sub_64EF60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char v5; // al
  unsigned int v6; // r13d
  _QWORD *v7; // rax
  size_t v9; // rax
  char *v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  int v13; // [rsp+Ch] [rbp-104h] BYREF
  _BYTE v14[173]; // [rsp+10h] [rbp-100h] BYREF
  char v15; // [rsp+BDh] [rbp-53h]

  if ( unk_4F04C50
    && (v4 = *(_QWORD *)(unk_4F04C50 + 32LL)) != 0
    && ((*(_BYTE *)(v4 + 198) & 0x10) != 0 || unk_4D04530 && (*(_BYTE *)(v4 + 193) & 2) != 0) )
  {
    v5 = *(_BYTE *)(a3 + 40);
    if ( v5 == 5 || (unsigned __int8)(v5 - 12) <= 2u )
    {
      v6 = 0x7FFFFFFF;
      sub_7C9660(a1);
      if ( word_4F06418[0] == 9 )
      {
LABEL_9:
        v7 = (_QWORD *)sub_7247C0(128);
        *(_QWORD *)(a3 + 64) = v7;
        *v7 = 0;
        v7[15] = 0;
        memset(
          (void *)((unsigned __int64)(v7 + 1) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)v7 - (((_DWORD)v7 + 8) & 0xFFFFFFF8) + 128) >> 3));
        sprintf(*(char **)(a3 + 64), "unroll %d", v6);
        return (char *)sub_7C96B0(1);
      }
      sub_6BA680(v14);
      if ( v15 == 1 )
      {
        v11 = 3625;
        if ( (int)sub_6210B0((__int64)v14, 0) <= 0 )
          goto LABEL_16;
        v12 = sub_620FD0((__int64)v14, &v13);
        if ( v13 || v12 > 0x7FFFFFFF )
        {
          v11 = 3624;
          goto LABEL_16;
        }
        v6 = v12;
      }
      else
      {
        if ( v15 != 12 )
        {
          v11 = 3626 - ((unsigned int)(v15 == 0) - 1);
LABEL_16:
          sub_684AA0(5, v11, a1 + 48);
          return (char *)sub_7C96B0(1);
        }
        if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
        {
LABEL_19:
          v11 = 3627;
          goto LABEL_16;
        }
      }
      if ( word_4F06418[0] == 9 )
        goto LABEL_9;
      goto LABEL_19;
    }
    return (char *)sub_684AA0(5, 3515, a1 + 48);
  }
  else
  {
    v9 = strlen(*(const char **)(a1 + 80));
    v10 = (char *)sub_7247C0(v9 + 1);
    *(_QWORD *)(a3 + 64) = v10;
    return strcpy(v10, *(const char **)(a1 + 80));
  }
}
