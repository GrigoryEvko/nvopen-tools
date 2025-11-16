// Function: sub_8FD6D0
// Address: 0x8fd6d0
//
__int64 __fastcall sub_8FD6D0(__int64 a1, const char *a2, _QWORD *a3)
{
  size_t v4; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // rcx
  __int64 v7; // rcx

  v4 = strlen(a2);
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v5 = v4;
  *(_QWORD *)a1 = a1 + 16;
  sub_2240E30(a1, v4 + a3[1]);
  if ( v5 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1, a2, v5, v6);
  sub_2241490(a1, *a3, a3[1], v7);
  return a1;
}
