// Function: sub_69A3A0
// Address: 0x69a3a0
//
__int64 __fastcall sub_69A3A0(char a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rsi
  _QWORD *v6; // r12
  unsigned int v7; // r13d
  __int64 v9; // rax
  __int64 v10; // rdx
  char v11; // r14
  bool v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // [rsp+8h] [rbp-238h] BYREF
  _BYTE v17[160]; // [rsp+10h] [rbp-230h] BYREF
  _BYTE v18[16]; // [rsp+B0h] [rbp-190h] BYREF
  char v19; // [rsp+C0h] [rbp-180h]
  __int64 v20; // [rsp+140h] [rbp-100h]

  sub_6E1DD0(&v16);
  v5 = v17;
  sub_6E1E00(5, v17, 0, 1);
  if ( (unsigned int)sub_8D2310(a2) || (v7 = sub_8D2310(a3)) != 0 )
  {
    v6 = 0;
    v7 = 0;
  }
  else
  {
    v6 = (_QWORD *)sub_68B9A0(a2);
    if ( v6 )
    {
      v9 = sub_68B9A0(a3);
      *v6 = v9;
      if ( v9 )
      {
        *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
        v10 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        v11 = *(_BYTE *)(v10 + 7);
        *(_BYTE *)(v10 + 7) = v11 & 0xF7;
        v12 = (v11 & 8) != 0;
        v5 = (_BYTE *)(*(_QWORD *)(v9 + 24) + 8LL);
        sub_6927A0((_QWORD *)(v6[3] + 8LL), (__int64)v5, (__int64)&dword_4F063F8, dword_4F06650[0], 1, (__int64)v18);
        v7 = !(*(_BYTE *)(qword_4D03C50 + 19LL) & 1);
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
        {
          if ( a1 == 45 )
          {
            if ( v19 != 1 || !(unsigned int)sub_731B40(v20, v5, v13, v14, v15) )
              goto LABEL_13;
          }
          else if ( a1 != 46 || v19 != 1 || !(unsigned int)sub_731CA0(v20) )
          {
            goto LABEL_13;
          }
          v7 = 0;
        }
LABEL_13:
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0]
                                                                            + 776LL * dword_4F04C64
                                                                            + 7)
                                                                 & 0xF7
                                                                 | (8 * v12);
      }
    }
  }
  sub_6E1990(v6);
  sub_6E2B30(v6, v5);
  sub_6E1DF0(v16);
  return v7;
}
