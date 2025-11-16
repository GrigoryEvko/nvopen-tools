// Function: sub_6726E0
// Address: 0x6726e0
//
__int64 __fastcall sub_6726E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdi
  char v5; // al
  unsigned int v6; // r12d
  char v7; // al
  char v8; // dl
  __int64 v10; // rax
  __int64 **v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rax
  _BYTE *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax

  if ( (*(_BYTE *)(a1 + 123) & 4) == 0 )
    return 0;
  v3 = *(_QWORD *)(a1 + 424);
  v4 = a2;
  if ( !dword_4D04490 && ((*(_BYTE *)(v3 + 131) & 8) == 0 || !unk_4D04484) )
    return 0;
  v5 = *(_BYTE *)(v3 + 133);
  if ( (v5 & 0x20) == 0 && !(v5 & 8 | *(_BYTE *)(v3 + 122) & 0xC) )
    return 0;
  v6 = unk_4F07758;
  if ( unk_4F07758 )
    return 0;
  if ( a2 )
    v4 = sub_8988D0(a2, 0);
  v7 = *(_BYTE *)(v3 + 122);
  if ( (v7 & 8) == 0 )
  {
    v8 = *(_BYTE *)(v3 + 133);
    if ( (v8 & 8) == 0 )
    {
      if ( !(v7 & 4 | v8 & 0x20) )
      {
        *(_QWORD *)(a1 + 272) = sub_72C930(v4);
        return v6;
      }
      if ( word_4F06418[0] != 77 )
      {
        sub_6851C0(3096, &dword_4F063F8);
        *(_QWORD *)(a1 + 272) = sub_72C930(3096);
        return v6;
      }
      if ( !sub_6725C0() )
      {
        v6 = 1;
        sub_643FD0(v3, v4);
        v15 = *(_QWORD *)(v3 + 368);
        *(_BYTE *)(a1 + 123) |= 0x20u;
        *(_QWORD *)(a1 + 368) = v15;
        *(_QWORD *)(a1 + 104) = *(_QWORD *)&dword_4F063F8;
        v16 = sub_72B6D0(a1 + 104, 0);
        *(_QWORD *)(a1 + 304) = v16;
        *(_QWORD *)(a1 + 272) = v16;
        return v6;
      }
      if ( v4 )
      {
        sub_6851C0(3098, a1 + 32);
        *(_QWORD *)(a1 + 104) = *(_QWORD *)&dword_4F063F8;
        v13 = sub_72B6D0(a1 + 104, 0);
        *(_BYTE *)(a1 + 125) |= 1u;
        *(_QWORD *)(a1 + 304) = v13;
        *(_QWORD *)(a1 + 272) = v13;
        return v6;
      }
      return 0;
    }
  }
  if ( word_4F06418[0] != 77 || sub_6725C0() )
    return 0;
  if ( (*(_BYTE *)(v3 + 122) & 8) != 0 )
    v10 = 776LL * unk_4F04C48;
  else
    v10 = 776LL * (int)dword_4F04C44;
  v11 = **(__int64 ****)(qword_4F04C68[0] + v10 + 408);
  if ( !v11 )
LABEL_41:
    BUG();
  while ( 1 )
  {
    v12 = v11[1];
    if ( *((_DWORD *)v12 + 14) == dword_4F06650[0] )
      break;
    v11 = (__int64 **)*v11;
    if ( !v11 )
      goto LABEL_41;
  }
  *(_QWORD *)(a1 + 272) = v12[11];
  if ( ((_BYTE)v11[7] & 0x10) == 0 )
    return 1;
  v6 = 1;
  if ( dword_4F04C64 != -1 )
  {
    v14 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    if ( (v14[7] & 1) != 0 && (dword_4F04C44 != -1 || (v14[6] & 6) != 0 || v14[4] == 12) )
    {
      sub_867130(v11[1], &dword_4F063F8, 0, 0);
      return 1;
    }
  }
  return v6;
}
