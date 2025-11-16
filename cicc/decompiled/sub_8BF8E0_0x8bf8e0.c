// Function: sub_8BF8E0
// Address: 0x8bf8e0
//
__int64 __fastcall sub_8BF8E0(unsigned __int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _WORD *v7; // r13
  unsigned int v8; // r12d
  int v9; // ebx
  __int64 v10; // rdx
  int v11; // r15d
  unsigned __int16 v12; // ax
  __int64 v13; // rax
  __int64 result; // rax
  unsigned int v15; // ecx
  char v16; // bl
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // [rsp+8h] [rbp-228h]
  __int64 v25; // [rsp+8h] [rbp-228h]
  __int64 v26; // [rsp+8h] [rbp-228h]
  __int64 v27; // [rsp+18h] [rbp-218h] BYREF
  _BYTE v28[528]; // [rsp+20h] [rbp-210h] BYREF

  v7 = (_WORD *)a1;
  v8 = a2;
  v9 = a2 & 4;
  v27 = *(_QWORD *)&dword_4F077C8;
  v10 = dword_4D04324;
  if ( dword_4D04324 )
  {
    a1 = (unsigned __int64)&dword_4F063F8;
    a2 = 875;
    sub_684AB0(&dword_4F063F8, 0x36Bu);
  }
  v11 = 0;
  v12 = word_4F06418[0];
  if ( word_4F06418[0] == 169 )
  {
    v11 = dword_4D04278;
    if ( dword_4D04278 )
    {
      v11 = 1;
      v27 = *(_QWORD *)&dword_4F063F8;
    }
    else if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
    {
      a2 = 1905;
      a1 = 7;
      sub_684AA0(7u, 0x771u, &dword_4F063F8);
    }
    else
    {
      a2 = 1073;
      a1 = 7;
      sub_684AA0(7u, 0x431u, &dword_4F063F8);
    }
    sub_7B8B50(a1, (unsigned int *)a2, v10, a4, a5, a6);
    v12 = word_4F06418[0];
  }
  if ( v12 == 160 || v9 )
  {
    if ( (unsigned __int16)sub_7BE840(0, 0) == 43 )
    {
      v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (v8 & 1) != 0 )
      {
        v25 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        sub_6851C0(0x1E1u, dword_4F07508);
        v13 = v25;
      }
      else if ( (v8 & 2) != 0 )
      {
        v26 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        sub_6851C0(0x146u, a3);
        v13 = v26;
      }
      if ( (*(_BYTE *)(v13 + 9) & 0xE) == 6 )
      {
        v24 = v13;
        sub_6851C0(0x320u, &dword_4F063F8);
        v15 = v9 != 0;
        v16 = *(_BYTE *)(v24 + 9) & 0x1E;
        *(_BYTE *)(v24 + 9) = *(_BYTE *)(v24 + 9) & 0xE1 | 4;
        result = sub_8BE160((__int64)v7, v11, &v27, v15, 0);
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) = *(_BYTE *)(qword_4F04C68[0]
                                                                            + 776LL * dword_4F04C64
                                                                            + 9)
                                                                 & 0xE1
                                                                 | v16;
      }
      else
      {
        return sub_8BE160((__int64)v7, v11, &v27, v9 != 0, 0);
      }
    }
    else
    {
      if ( v11 )
        sub_6851C0(0x42Cu, &v27);
      sub_8BF710((__int64)v28, v8, a3);
      return 0;
    }
  }
  else
  {
    v17 = qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    ++*(_BYTE *)(v17 + 82);
    sub_6851D0(0x42Bu);
    v18 = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 82LL);
    --*(_BYTE *)(v18 + 83);
    v19 = word_4F06418[0];
    if ( word_4F06418[0] == 74 )
    {
      if ( (unsigned __int16)sub_7BE840(0, 0) == 75 )
        sub_7B8B50(0, 0, v20, v21, v22, v23);
      v19 = word_4F06418[0];
    }
    *v7 = v19;
    return 0;
  }
  return result;
}
