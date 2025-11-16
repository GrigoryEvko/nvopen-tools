// Function: sub_6D7740
// Address: 0x6d7740
//
_BYTE *__fastcall sub_6D7740(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  char i; // dl
  _BYTE *v8; // rsi
  _BYTE *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  char v17; // al
  char v19[160]; // [rsp+0h] [rbp-220h] BYREF
  _QWORD v20[2]; // [rsp+A0h] [rbp-180h] BYREF
  char v21; // [rsp+B0h] [rbp-170h]

  sub_7B8B50(a1, a2, a3, a4);
  if ( word_4F06418[0] != 27 )
  {
    v9 = 0;
    sub_6851C0(0x7Du, &dword_4F063F8);
    return v9;
  }
  sub_7B8B50(a1, a2, v4, v5);
  sub_6E1E00(3, v19, 0, 0);
  sub_69ED20((__int64)v20, 0, 0, 0);
  sub_6F69D0(v20, 0);
  sub_6E2B30(v20, 0);
  if ( !v21 )
    goto LABEL_16;
  v6 = v20[0];
  for ( i = *(_BYTE *)(v20[0] + 140LL); i == 12; i = *(_BYTE *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  if ( !i )
  {
LABEL_16:
    v9 = 0;
    sub_6E6870(v20);
    sub_684AA0(7u, 0xE05u, &dword_4F063F8);
    return v9;
  }
  sub_6E6B60(v20, 0);
  if ( v21 != 2 )
  {
    v9 = 0;
    sub_6E68E0(28, v20);
    return v9;
  }
  v8 = (_BYTE *)sub_724D80(0);
  v9 = v8;
  sub_6F4950(v20, v8, v10, v11, v12, v13);
  if ( word_4F06418[0] != 28 )
  {
    v9 = 0;
    sub_6851C0(0x12u, &dword_4F063F8);
    return v9;
  }
  sub_7B8B50(v20, v8, v14, v15);
  if ( v8 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v16 + 6) & 6) == 0 && *(_BYTE *)(v16 + 4) != 12 )
      {
        v17 = v8[173];
        if ( v17 == 7 )
        {
          if ( (v8[192] & 2) != 0 )
            return v9;
          goto LABEL_15;
        }
        if ( v17 != 6 || v8[176] )
        {
LABEL_15:
          v9 = 0;
          sub_684AA0(7u, 0xE07u, &dword_4F063F8);
        }
      }
    }
  }
  return v9;
}
