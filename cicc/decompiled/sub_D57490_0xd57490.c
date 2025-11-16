// Function: sub_D57490
// Address: 0xd57490
//
__int64 __fastcall sub_D57490(__int64 a1, __int64 a2, __m128i a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rcx
  char *v5; // rax
  __int64 v6; // rdx

  v3 = sub_D573D0(*(_QWORD **)(a2 + 32), *(_QWORD *)(a2 + 40));
  if ( v4 == v3 )
    return 0;
  v5 = (char *)sub_BD5D20(*(_QWORD *)(*v3 + 72LL));
  if ( !sub_BC63A0(v5, v6) )
    return 0;
  sub_D4BD90(a2, *(char **)(a1 + 176), a1 + 184, a3);
  return 0;
}
