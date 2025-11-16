// Function: sub_6913B0
// Address: 0x6913b0
//
__int64 __fastcall sub_6913B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v4; // bx
  __int64 v5; // r12
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v9; // rdx
  __int64 *v10; // rax
  char i; // dl
  __int64 v12; // r12
  bool v13; // al
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rdx
  __int64 *v18; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v19[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = word_4F06418[0];
  v5 = (__int64)*(&off_4B6DFA0 + word_4F06418[0]);
  sub_7B8B50(a1, a2, word_4F06418[0], a4);
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v19[0] = *(_QWORD *)&dword_4F063F8;
  sub_65CD60(&v18);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  sub_7BE280(28, 18, 0, 0);
  if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v9 = 776LL * dword_4F04C64 + qword_4F04C68[0];
      if ( (*(_BYTE *)(v9 + 6) & 6) == 0 && *(_BYTE *)(v9 + 4) != 12 )
      {
        v10 = v18;
        for ( i = *((_BYTE *)v18 + 140); i == 12; i = *((_BYTE *)v10 + 140) )
          v10 = (__int64 *)v10[20];
        v18 = v10;
        if ( (unsigned __int8)(i - 9) <= 2u )
          goto LABEL_14;
        v7 = 2413;
        sub_6851C0(0x96Du, v19);
        return sub_72C930(v7);
      }
    }
  }
  else if ( dword_4F04C44 == -1 )
  {
    v7 = 2415;
    sub_6851A0(0x96Fu, v19, v5);
    return sub_72C930(v7);
  }
  v6 = *v18;
  if ( !*v18 || !(unsigned int)sub_8D3D40(v18) || (*(_BYTE *)(v6 + 81) & 0x40) == 0 )
  {
    v7 = 2414;
    sub_6851C0(0x96Eu, v19);
    return sub_72C930(v7);
  }
LABEL_14:
  v12 = sub_7259C0(12);
  v13 = 1;
  if ( dword_4F04C44 == -1 )
  {
    v17 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v17 + 6) & 6) == 0 )
      v13 = *(_BYTE *)(v17 + 4) == 12;
  }
  *(_BYTE *)(v12 + 186) = (8 * v13) | *(_BYTE *)(v12 + 186) & 0xF7;
  v14 = v18;
  *(_BYTE *)(v12 + 184) = (v4 == 250) + 11;
  v15 = sub_8674D0(v14, v19, v4 == 250);
  v16 = v18;
  *(_QWORD *)(v12 + 160) = v15;
  *(_QWORD *)(*(_QWORD *)(v12 + 168) + 40LL) = v16;
  return v12;
}
