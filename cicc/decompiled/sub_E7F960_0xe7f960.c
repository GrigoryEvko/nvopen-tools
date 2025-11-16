// Function: sub_E7F960
// Address: 0xe7f960
//
unsigned __int64 __fastcall sub_E7F960(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned __int8 **v6; // rbx
  unsigned __int64 result; // rax
  unsigned __int8 **i; // r13
  unsigned __int8 *v9; // rsi

  sub_E8B980();
  v5 = *(_QWORD *)(a1 + 288);
  v6 = *(unsigned __int8 ***)(v5 + 72);
  result = 3LL * *(unsigned int *)(v5 + 80);
  for ( i = &v6[result]; i != v6; result = sub_E7E6A0(a1, v9, v1, v2, v3, v4) )
  {
    v9 = *v6;
    v6 += 3;
  }
  return result;
}
