// Function: sub_F05F90
// Address: 0xf05f90
//
__int64 __fastcall sub_F05F90(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx

  v2 = a2[1];
  v3 = *a2;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( v3 != v2 )
  {
    v4 = v3;
    v5 = ((v2 - v3) >> 5) - 1;
    do
    {
      v5 += *(_QWORD *)(v4 + 8);
      v4 += 32;
    }
    while ( v2 != v4 );
    v6 = v3 + 32;
    sub_2240E30(a1, v5);
    sub_2241490(a1, *(_QWORD *)(v6 - 32), *(_QWORD *)(v6 - 24), v7);
    while ( v2 != v6 )
    {
      if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v6 += 32;
      sub_2241490(a1, ",", 1, v8);
      sub_2241490(a1, *(_QWORD *)(v6 - 32), *(_QWORD *)(v6 - 24), v9);
    }
  }
  return a1;
}
