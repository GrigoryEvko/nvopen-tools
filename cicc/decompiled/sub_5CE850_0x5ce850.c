// Function: sub_5CE850
// Address: 0x5ce850
//
__int64 __fastcall sub_5CE850(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 i; // rax
  char *v6; // rax

  v2 = *(_QWORD *)(a1 + 48);
  if ( unk_4D04964 )
    sub_684AA0(5, 2480, a1 + 56);
  v3 = unk_4F04C68 + 776LL * unk_4F04C64;
  if ( *(_BYTE *)(v3 + 4) == 6 )
  {
    if ( v2 && (*(_BYTE *)(v2 + 8) & 8) != 0 )
    {
      sub_5CCAE0(8u, a1);
      return a2;
    }
    else
    {
      for ( i = *(_QWORD *)(v3 + 208); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 181LL) |= 0x20u;
      return a2;
    }
  }
  else
  {
    v6 = sub_5C79F0(a1);
    sub_6851A0(1848, a1 + 56, v6);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
}
