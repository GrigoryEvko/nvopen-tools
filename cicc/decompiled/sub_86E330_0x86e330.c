// Function: sub_86E330
// Address: 0x86e330
//
__int64 __fastcall sub_86E330(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  sub_854980(0, a1);
  v2 = a1;
  v3 = 8;
  sub_86D170(8, v2, 0, 0, v4, v5);
  if ( dword_4F077C4 == 2 )
  {
    v2 = *(_QWORD *)(a1 + 72);
    v3 = 19;
    sub_733780(0x13u, v2, 0, 5, 0);
  }
  *(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 202LL) |= 0x10u;
  if ( !dword_4D048B8 )
  {
    v2 = (__int64)&dword_4F063F8;
    v3 = 540;
    sub_6851C0(0x21Cu, &dword_4F063F8);
    return sub_7B8B50(v3, (unsigned int *)v2, v6, v7, v8, v9);
  }
  if ( dword_4F04C58 != -1
    && (*(_BYTE *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216) + 198LL) & 0x10) != 0 )
  {
    v2 = (__int64)&dword_4F063F8;
    v3 = 3517;
    sub_6851C0(0xDBDu, &dword_4F063F8);
    return sub_7B8B50(v3, (unsigned int *)v2, v6, v7, v8, v9);
  }
  if ( dword_4D04324 )
  {
    v3 = (unsigned __int64)&dword_4F063F8;
    v2 = 876;
    sub_684AB0(&dword_4F063F8, 0x36Cu);
    if ( !sub_86D9F0() )
      return sub_7B8B50(v3, (unsigned int *)v2, v6, v7, v8, v9);
    goto LABEL_10;
  }
  if ( sub_86D9F0() )
  {
LABEL_10:
    v2 = (__int64)&dword_4F063F8;
    v3 = 1229;
    sub_6851C0(0x4CDu, &dword_4F063F8);
  }
  return sub_7B8B50(v3, (unsigned int *)v2, v6, v7, v8, v9);
}
