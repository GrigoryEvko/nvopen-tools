// Function: sub_31F17D0
// Address: 0x31f17d0
//
void __fastcall sub_31F17D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax

  v4 = sub_31DE680(a1, *(_QWORD *)(a2 + 24), a3);
  sub_EA12C0(v4, a3, *(_BYTE **)(a1 + 208));
  sub_31DCB40(a1, *(unsigned int *)(a2 + 8) | (unsigned __int64)((__int64)*(int *)(a2 + 32) << 32), a3);
}
