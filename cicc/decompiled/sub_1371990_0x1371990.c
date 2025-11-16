// Function: sub_1371990
// Address: 0x1371990
//
__int64 __fastcall sub_1371990(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v2 )
    j_j___libc_free_0(v2, v3 - v2);
  v4 = *(_QWORD **)(a1 + 40);
  while ( (_QWORD *)(a1 + 40) != v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    j_j___libc_free_0(v5, 40);
  }
  v6 = *(_QWORD *)(a1 + 64);
  v7 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 48) = v4;
  *(_QWORD *)(a1 + 40) = v4;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  if ( v6 )
    j_j___libc_free_0(v6, v7 - v6);
  result = sub_1371900((_QWORD **)(a1 + 88));
  *(_QWORD *)(a1 + 96) = a1 + 88;
  *(_QWORD *)(a1 + 88) = a1 + 88;
  *(_QWORD *)(a1 + 104) = 0;
  return result;
}
