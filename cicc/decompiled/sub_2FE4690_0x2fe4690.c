// Function: sub_2FE4690
// Address: 0x2fe4690
//
__int64 __fastcall sub_2FE4690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // bl
  __int64 v5; // rdx
  __int64 v6; // rax
  char *v7; // rsi
  size_t v8; // rdx
  __int64 v9; // rdx
  __int16 v10; // ax
  unsigned __int16 v11; // bx
  _WORD *v12; // rdx
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  _QWORD v16[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = a2;
  v16[0] = a3;
  v16[1] = a4;
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) > 0xD3u )
    {
LABEL_3:
      v5 = a1 + 16;
      v6 = 0;
      *(_QWORD *)a1 = a1 + 16;
      goto LABEL_4;
    }
  }
  else if ( !(unsigned __int8)sub_30070B0(v16, a2, a3) )
  {
    goto LABEL_3;
  }
  v5 = a1 + 16;
  v6 = 4;
  *(_DWORD *)(a1 + 16) = 761488758;
  *(_QWORD *)a1 = a1 + 16;
LABEL_4:
  *(_QWORD *)(a1 + 8) = v6;
  v7 = "sqrt";
  *(_BYTE *)(v5 + v6) = 0;
  v8 = 4LL - (v4 == 0);
  if ( !v4 )
    v7 = "div";
  if ( v8 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
    goto LABEL_29;
  sub_2241490((unsigned __int64 *)a1, v7, v8);
  v10 = v16[0];
  v11 = v16[0];
  if ( LOWORD(v16[0]) )
  {
    v12 = (_WORD *)((unsigned int)LOWORD(v16[0]) - 17);
    if ( (unsigned __int16)(LOWORD(v16[0]) - 17) <= 0xD3u )
    {
      v12 = word_4456580;
      v10 = word_4456580[LOWORD(v16[0]) - 1];
    }
    if ( v10 == 13 )
    {
LABEL_11:
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, "d", 1u);
        return a1;
      }
LABEL_29:
      sub_4262D8((__int64)"basic_string::append");
    }
  }
  else
  {
    if ( !(unsigned __int8)sub_30070B0(v16, v7, v9) )
      goto LABEL_16;
    if ( (unsigned __int16)sub_3009970(v16) == 13 )
      goto LABEL_11;
  }
  v11 = v16[0];
LABEL_16:
  v14 = v11;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
      v14 = word_4456580[v11 - 1];
  }
  else
  {
    if ( !(unsigned __int8)sub_30070B0(v16, v7, v12) )
    {
      v15 = *(_QWORD *)(a1 + 8);
      goto LABEL_24;
    }
    v14 = sub_3009970(v16);
  }
  v15 = *(_QWORD *)(a1 + 8);
  if ( v14 == 11 )
  {
    if ( v15 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_29;
    sub_2241490((unsigned __int64 *)a1, "h", 1u);
    return a1;
  }
LABEL_24:
  if ( v15 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_29;
  sub_2241490((unsigned __int64 *)a1, "f", 1u);
  return a1;
}
