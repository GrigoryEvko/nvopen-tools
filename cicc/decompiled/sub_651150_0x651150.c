// Function: sub_651150
// Address: 0x651150
//
_BOOL8 __fastcall sub_651150(int a1)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rsi
  unsigned __int16 v5; // ax
  _BOOL8 v6; // r12
  unsigned int v8; // r15d
  __int64 v9; // rax
  int v10; // ecx
  char v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rdx
  _BYTE v18[80]; // [rsp+10h] [rbp-50h] BYREF

  sub_7ADF70(v18, 0);
  sub_7AE360(v18);
  sub_7B8B50(v18, 0, v2, v3);
  v4 = 0;
  v5 = word_4F06418[0];
  if ( word_4F06418[0] == 43 )
  {
    v8 = dword_4F06650[0];
    sub_7BDB60(1);
    v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v10 = *(unsigned __int8 *)(v9 + 12);
    ++*(_QWORD *)(v9 + 632);
    v11 = v10;
    v12 = v10 | 4u;
    *(_BYTE *)(v9 + 12) = v12;
    sub_7B8B50(1, 0, qword_4F04C68, v12);
    v13 = sub_7BF3A0(0, 0);
    sub_725130(v13);
    v14 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v15 = 4 * (unsigned int)((v11 & 4) != 0);
    v16 = *(unsigned __int8 *)(v14 + 12);
    --*(_QWORD *)(v14 + 632);
    v17 = (unsigned int)v15 | v16 & 0xFFFFFFFB;
    *(_BYTE *)(v14 + 12) = v17;
    if ( word_4F06418[0] == 44 )
      sub_7B8B50(v13, 0, v17, v15);
    sub_7BDC00();
    sub_7AE700(unk_4F061C0 + 24LL, v8, unk_4F0664C, 1, v18);
    v5 = word_4F06418[0];
    v4 = word_4F06418[0] != 9;
  }
  LODWORD(v6) = 1;
  if ( v5 != 77 )
  {
    if ( a1 && v5 == 185 )
    {
      v4 = (unsigned int)v4;
      v6 = sub_6510D0();
    }
    else
    {
      LODWORD(v6) = 0;
    }
  }
  sub_7BBF80(v18, v4);
  return v6;
}
