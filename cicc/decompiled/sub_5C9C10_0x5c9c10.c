// Function: sub_5C9C10
// Address: 0x5c9c10
//
__int64 __fastcall sub_5C9C10(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  char *v5; // rax

  if ( unk_4D04964 )
    sub_684AA0(5, 2480, a1 + 56);
  if ( *(_BYTE *)(unk_4F04C68 + 776LL * unk_4F04C64 + 4) == 6 )
  {
    v3 = *(_QWORD *)(a1 + 48);
    if ( (*(_BYTE *)(v3 + 129) & 1) == 0 )
      sub_6851C0(1455, v3 + 48);
    return a2;
  }
  else
  {
    v5 = sub_5C79F0(a1);
    sub_6851A0(1848, a1 + 56, v5);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
}
