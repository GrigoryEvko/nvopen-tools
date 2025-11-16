// Function: sub_69A560
// Address: 0x69a560
//
__int64 __fastcall sub_69A560(_BYTE *a1, __int64 a2, char a3)
{
  unsigned int v6; // r15d
  _BYTE *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r10
  int v13; // r8d
  __int64 v14; // rdi
  __int64 v15; // r13
  int v16; // eax
  __int64 v17; // r8
  __int64 v18; // rdx
  char v19; // bl
  bool v20; // bl
  __int64 v21; // r10
  __int64 v22; // r12
  __int64 v23; // rax
  char v24; // al
  unsigned int v25; // [rsp+8h] [rbp-E8h]
  __int64 v26; // [rsp+8h] [rbp-E8h]
  unsigned int v27; // [rsp+14h] [rbp-DCh] BYREF
  __int64 v28; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE v29[208]; // [rsp+20h] [rbp-D0h] BYREF

  v6 = sub_8D32E0(a1);
  if ( v6 )
  {
    sub_6E1DD0(&v28);
    v8 = v29;
    sub_6E1E00(5, v29, 0, 1);
    sub_7296C0(&v27);
    v9 = sub_68B9A0(a2);
    v10 = v9;
    if ( !v9 )
    {
      v6 = 0;
      goto LABEL_13;
    }
    v11 = *(_QWORD *)(v9 + 24);
    v12 = v11 + 8;
    if ( a3 == 66 )
    {
      v17 = 0x40000;
      v15 = 0;
    }
    else
    {
      v13 = 263168;
      v14 = a2;
      if ( a3 != 109 )
        v13 = 0x40000;
      v15 = 0;
      v25 = v13;
      v16 = sub_8D2FB0(v14);
      v17 = v25;
      v12 = v11 + 8;
      if ( !v16 )
      {
        v15 = *(_QWORD *)(v11 + 152);
        *(_BYTE *)(v15 + 25) &= 0xFCu;
        *(_BYTE *)(v11 + 25) = 2;
      }
    }
    v8 = a1;
    *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
    v18 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v19 = *(_BYTE *)(v18 + 7);
    *(_BYTE *)(v18 + 7) = v19 & 0xF7;
    v26 = v12;
    v20 = (v19 & 8) != 0;
    sub_842520(v12, a1, 0, 1, v17, 0);
    v21 = v26;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 && *(_BYTE *)(v11 + 24) == 1 )
    {
      v22 = *(_QWORD *)(v11 + 152);
      v8 = 0;
      v6 = 1;
      v23 = sub_844780(v22, 0);
      v21 = v26;
      if ( v23 )
        goto LABEL_12;
      if ( v15 )
      {
        if ( v15 == v22 )
          goto LABEL_12;
        if ( *(_BYTE *)(v22 + 24) == 1 )
        {
          v24 = *(_BYTE *)(v22 + 56);
          if ( v24 == 8 || v24 == 14 )
          {
            v6 = *(_QWORD *)(v22 + 72) == v15;
            goto LABEL_12;
          }
        }
      }
    }
    v6 = 0;
LABEL_12:
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                             & 0xF7
                                                             | (8 * v20);
    sub_6E4710(v21);
LABEL_13:
    sub_729730(v27);
    sub_6E1990(v10);
    sub_6E2B30(v10, v8);
    sub_6E1DF0(v28);
  }
  return v6;
}
