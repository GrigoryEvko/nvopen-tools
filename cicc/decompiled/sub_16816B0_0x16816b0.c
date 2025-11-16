// Function: sub_16816B0
// Address: 0x16816b0
//
__int64 __fastcall sub_16816B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rcx

  v2 = a2[1];
  v3 = *a2;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( v3 != v2 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    v5 = v2 - 32 - v3;
    v6 = v2 - v3;
    v7 = v3 + 32;
    sub_2240E30(a1, v4 + (v6 >> 5) - 1 + v4 * (v5 >> 5));
    sub_2241490(a1, *(_QWORD *)(v7 - 32), *(_QWORD *)(v7 - 24), v8);
    while ( v2 != v7 )
    {
      if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v7 += 32;
      sub_2241490(a1, ",", 1, v9);
      sub_2241490(a1, *(_QWORD *)(v7 - 32), *(_QWORD *)(v7 - 24), v10);
    }
  }
  return a1;
}
