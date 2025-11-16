// Function: sub_5C7D80
// Address: 0x5c7d80
//
__int64 __fastcall sub_5C7D80(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  char *v5; // rax
  char *v6; // rax

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
  if ( a3 != 11 )
  {
    if ( a3 != 7 )
      sub_721090(a1);
    if ( unk_4F077A8 <= 0x9D07u && *(_BYTE *)(a2 + 136) > 1u )
    {
      v6 = sub_5C79F0(a1);
      sub_6851A0(1407, a1 + 56, v6);
      *(_BYTE *)(a1 + 8) = 0;
    }
    else
    {
      if ( *(_BYTE *)(a1 + 8) == 26 )
        *(_BYTE *)(a2 + 168) |= 0x20u;
      sub_5C7230(*(_QWORD *)a2, 0, *(_QWORD *)(v3 + 184), (_QWORD *)(a1 + 56));
    }
    return a2;
  }
  if ( unk_4F077A8 > 0x9C3Fu && unk_4F04C50 )
  {
    v5 = sub_5C79F0(a1);
    sub_684B10(1664, a1 + 56, v5);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  if ( (*(_BYTE *)(a2 + 201) & 1) != 0 )
  {
    sub_6851C0(2537, a1 + 56);
    return a2;
  }
  *(_BYTE *)(a2 + 202) &= ~4u;
  if ( *(_BYTE *)(a1 + 8) == 26 )
    *(_BYTE *)(a2 + 200) |= 0x80u;
  sub_5C7230(*(_QWORD *)a2, 0, *(_QWORD *)(v3 + 184), (_QWORD *)(a1 + 56));
  return a2;
}
