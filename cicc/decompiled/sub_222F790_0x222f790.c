// Function: sub_222F790
// Address: 0x222f790
//
__int64 __fastcall sub_222F790(_QWORD *a1, __int64 a2)
{
  signed __int64 *v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 result; // rax

  v3 = &qword_4FD6808;
  v4 = sub_22091A0(&qword_4FD6808);
  v5 = *(_QWORD *)(*a1 + 8LL);
  if ( *(_QWORD *)(*a1 + 16LL) <= v4 || (v3 = *(signed __int64 **)(v5 + 8 * v4)) == 0 )
    sub_426219(v3, a2, *a1, v5);
  result = sub_2252480(v3, &`typeinfo for'std::locale::facet, &`typeinfo for'std::ctype<char>, 0);
  if ( !result )
    sub_426611();
  return result;
}
