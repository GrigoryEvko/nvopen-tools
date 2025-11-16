// Function: sub_6911B0
// Address: 0x6911b0
//
__int64 __fastcall sub_6911B0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r12
  __int64 v4; // rdx
  unsigned __int16 v5; // r13
  char v7; // dl
  __int64 v8; // rax
  __int64 v9; // rax
  bool v10; // r14
  __int64 v11; // rbx
  char v12; // al
  __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = sub_72C930(a1);
  v4 = dword_4D044B0;
  v5 = word_4F06418[0];
  if ( !dword_4D044B0 )
  {
    a1 = 1543;
    a2 = &dword_4F063F8;
    sub_6851A0(0x607u, &dword_4F063F8, (__int64)*(&off_4B6DFA0 + word_4F06418[0]));
  }
  sub_7B8B50(a1, a2, v4, v2);
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v16[0] = *(_QWORD *)&dword_4F063F8;
  sub_65CD60(&v15);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  sub_7BE280(28, 18, 0, 0);
  if ( dword_4D044B0 )
  {
    if ( (unsigned int)sub_8D3D40(v15) )
    {
      v3 = v15;
    }
    else
    {
      v14 = sub_687A50(v5);
      v3 = sub_8D5290(v15, v14, v16, 1);
    }
    v7 = *(_BYTE *)(v3 + 140);
    if ( v7 == 12 )
    {
      v8 = v3;
      do
      {
        v8 = *(_QWORD *)(v8 + 160);
        v7 = *(_BYTE *)(v8 + 140);
      }
      while ( v7 == 12 );
    }
    if ( v7 )
    {
      if ( dword_4F04C44 != -1
        || (v9 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v9 + 6) & 6) != 0)
        || (v10 = 0, *(_BYTE *)(v9 + 4) == 12) )
      {
        v10 = (unsigned int)sub_8DBE70(v3) != 0;
      }
      v11 = sub_7259C0(12);
      v12 = sub_687A50(v5);
      *(_QWORD *)(v11 + 160) = v3;
      v3 = v11;
      *(_BYTE *)(v11 + 184) = v12;
      v13 = *(_QWORD *)(v11 + 168);
      *(_BYTE *)(v11 + 186) = (8 * v10) | *(_BYTE *)(v11 + 186) & 0xF7;
      *(_QWORD *)(v13 + 40) = v15;
    }
  }
  return v3;
}
