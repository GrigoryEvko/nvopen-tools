// Function: sub_6472A0
// Address: 0x6472a0
//
__int64 __fastcall sub_6472A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  unsigned __int8 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 i; // rax
  __int64 v15; // rax
  __int64 v16; // rax

  v4 = a1;
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(_BYTE *)(a1 + 268);
  if ( (*(_BYTE *)(v5 + 140) & 0xFB) == 8 )
  {
    a1 = *(_QWORD *)(a1 + 272);
    if ( (unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2) )
    {
      if ( dword_4F077BC && qword_4F077A8 > 0x9C41u )
      {
        while ( *(_BYTE *)(v5 + 140) == 12 )
          v5 = *(_QWORD *)(v5 + 160);
        a1 = 1566;
        sub_684B30(1566, v4 + 24);
      }
      else
      {
        a1 = 1565;
        sub_684B30(1565, v4 + 24);
      }
    }
  }
  v7 = (unsigned int)dword_4F04C34;
  if ( dword_4F04C64 == dword_4F04C34 )
  {
    if ( v6 <= 1u )
    {
      sub_6851C0(365, dword_4F07508);
      v7 = (unsigned int)dword_4F04C34;
    }
    else if ( v6 != 2 )
    {
      sub_6851C0(149, dword_4F07508);
      v7 = (unsigned int)dword_4F04C34;
    }
    v15 = sub_735FB0(v5, 2, v7, a4);
    *(_BYTE *)(v15 + 172) |= 2u;
    v10 = v15;
    *(_BYTE *)(v15 + 137) = *(_BYTE *)(v4 + 268);
    *(_QWORD *)(v15 + 64) = *(_QWORD *)&dword_4F063F8;
    v16 = sub_87FA00(7, &dword_4F063F8, *(unsigned int *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C));
    *(_QWORD *)(v16 + 88) = v10;
    v12 = v16;
    sub_877E90(v16, v10);
  }
  else
  {
    if ( v6 == 1 )
    {
      sub_6851C0(365, dword_4F07508);
      v8 = 3;
    }
    else if ( v6 <= 1u )
    {
      v8 = 3;
    }
    else
    {
      if ( v6 > 3u && v6 != 5 )
        sub_721090(a1);
      v8 = v6;
    }
    v9 = sub_735FB0(v5, v8, (unsigned int)dword_4F04C5C, a4);
    *(_BYTE *)(v9 + 172) |= 2u;
    v10 = v9;
    *(_BYTE *)(v9 + 137) = *(_BYTE *)(v4 + 268);
    *(_QWORD *)(v9 + 64) = *(_QWORD *)&dword_4F063F8;
    v11 = sub_87FA00(7, &dword_4F063F8, *(unsigned int *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C));
    *(_QWORD *)(v11 + 88) = v10;
    v12 = v11;
    if ( unk_4F04C50 )
      *(_BYTE *)(v10 + 89) |= 1u;
  }
  for ( i = v5; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *(_BYTE *)(i + 143) |= 8u;
  *(_QWORD *)(v10 + 256) = v5;
  if ( !dword_4F04C3C )
    sub_8699D0(v10, 7, 0);
  if ( dword_4D04434 )
    sub_63BB10(v12, v4 + 24);
  if ( (dword_4D0488C
     || word_4D04898 && (_DWORD)qword_4F077B4 && qword_4F077A0 > 0x765Bu && (unsigned int)sub_729F80(dword_4F063F8))
    && unk_4F04C50
    && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 2) != 0 )
  {
    sub_646FB0(v10, v4 + 32);
  }
  sub_5F40E0(v12, 0);
  return sub_86F690(v12);
}
