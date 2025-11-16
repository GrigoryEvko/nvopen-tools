// Function: sub_39786C0
// Address: 0x39786c0
//
__int64 __fastcall sub_39786C0(__int64 a1, __int64 a2)
{
  const char *v2; // rax
  __int64 v3; // rdx
  char v4; // al
  unsigned int v5; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  const char *v9; // rax
  __int64 v10; // rdx
  const char *v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // r13
  __int64 v14; // rax
  _BYTE *v15; // rdx
  _BYTE *v16; // r13
  __int64 v17; // rax
  _BYTE *v18; // rdx

  v2 = sub_1649960(a2);
  if ( v3 == 9 && *(_QWORD *)v2 == 0x6573752E6D766C6CLL && v2[8] == 100 )
  {
    v5 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 240) + 308LL);
    if ( (_BYTE)v5 )
    {
      sub_396F250(a1, *(_QWORD *)(a2 - 24));
      return v5;
    }
  }
  else if ( (*(_BYTE *)(a2 + 34) & 0x20) == 0
         || (v7 = sub_15E61A0(a2), v8 != 13)
         || *(_QWORD *)v7 != 0x74656D2E6D766C6CLL
         || *(_DWORD *)(v7 + 8) != 1952539745
         || *(_BYTE *)(v7 + 12) != 97 )
  {
    v4 = *(_BYTE *)(a2 + 32) & 0xF;
    if ( v4 != 1 )
    {
      v5 = 0;
      if ( v4 == 6 )
      {
        v9 = sub_1649960(a2);
        if ( v10 == 17
          && !(*(_QWORD *)v9 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v9 + 1) ^ 0x726F74635F6C6162LL)
          && v9[16] == 115 )
        {
          v13 = *(_BYTE **)(a2 - 24);
          v14 = sub_1632FA0(*(_QWORD *)(a2 + 40));
          v15 = v13;
          v5 = 1;
          sub_3978030((_QWORD **)a1, v14, v15, 1);
        }
        else
        {
          v11 = sub_1649960(a2);
          if ( v12 != 17
            || *(_QWORD *)v11 ^ 0x6F6C672E6D766C6CLL | *((_QWORD *)v11 + 1) ^ 0x726F74645F6C6162LL
            || v11[16] != 115 )
          {
            sub_16BD130("unknown special variable", 1u);
          }
          v16 = *(_BYTE **)(a2 - 24);
          v17 = sub_1632FA0(*(_QWORD *)(a2 + 40));
          v18 = v16;
          v5 = 1;
          sub_3978030((_QWORD **)a1, v17, v18, 0);
        }
      }
      return v5;
    }
  }
  return 1;
}
