// Function: sub_318D6E0
// Address: 0x318d6e0
//
__int64 __fastcall sub_318D6E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 i; // r12
  __int64 v5; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  result = 9LL * *(unsigned int *)(a1 + 16);
  for ( i = v2 + 72LL * *(unsigned int *)(a1 + 16); v2 != i; result = sub_BD72D0(v5, a2) )
  {
    v5 = *(_QWORD *)(v2 + 64);
    v2 += 72;
  }
  return result;
}
