// Function: sub_889E70
// Address: 0x889e70
//
__int64 __fastcall sub_889E70(unsigned __int64 a1)
{
  int v2; // r14d
  unsigned __int8 v3; // di
  int v4; // r15d
  unsigned __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int16 v9; // dx
  char *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // r9
  __int64 v15; // r13
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 *v19; // [rsp+8h] [rbp-38h]

  v2 = dword_4F04C5C;
  sub_8896D0(a1);
  v3 = *(_BYTE *)(a1 + 74);
  if ( v3 == 8 || !(unsigned int)sub_889000(v3, *(_WORD *)(a1 + 76), 1) )
    return 0;
  sub_860350();
  v4 = 0;
  dword_4F04C5C = 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) & 0xE) != 6 )
  {
    v4 = 1;
    sub_85F8B0(3);
  }
  v5 = *(_WORD *)(a1 + 76);
  v6 = *(unsigned __int8 *)(a1 + 74);
  if ( (_BYTE)v6 == 9 )
  {
    v17 = 32LL * v5 + 77930624;
    v18 = sub_888BD0((&off_4A52090)[4 * v5], (__int64 *)&dword_4F063F8);
    v9 = *(_WORD *)(v17 + 24);
    v8 = v18;
  }
  else
  {
    v7 = qword_4A598E0[v6] + 16LL * v5;
    v8 = *(_QWORD *)(unk_4D03FB8 + 8LL * *(unsigned __int16 *)(v7 + 10));
    if ( !v8 )
    {
      v19 = (__int64 *)(unk_4D03FB8 + 8LL * *(unsigned __int16 *)(v7 + 10));
      v8 = sub_888BD0(off_4AE4E20[*(unsigned __int16 *)(v7 + 10)], (__int64 *)&dword_4F063F8);
      *v19 = v8;
    }
    v9 = *(_WORD *)(v7 + 12);
  }
  v10 = *(char **)(a1 + 8);
  v15 = sub_889970(v10, v8, v9, 0);
  if ( v4 )
    sub_85F950();
  dword_4F04C5C = v2;
  sub_863FC0((__int64)v10, v8, v11, v12, v13, v14);
  return v15;
}
