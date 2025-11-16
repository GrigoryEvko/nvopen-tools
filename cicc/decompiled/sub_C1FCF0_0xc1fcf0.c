// Function: sub_C1FCF0
// Address: 0xc1fcf0
//
__int64 __fastcall sub_C1FCF0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  _QWORD *v4; // rdi

  v2 = *(_QWORD *)(a1 + 136);
  while ( v2 )
  {
    v3 = v2;
    sub_C1F230(*(_QWORD **)(v2 + 24));
    v4 = *(_QWORD **)(v2 + 56);
    v2 = *(_QWORD *)(v2 + 16);
    sub_C1F480(v4);
    j_j___libc_free_0(v3, 88);
  }
  return sub_C1EF60(*(_QWORD **)(a1 + 88));
}
