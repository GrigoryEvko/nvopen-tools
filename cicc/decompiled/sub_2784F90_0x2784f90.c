// Function: sub_2784F90
// Address: 0x2784f90
//
__int64 __fastcall sub_2784F90(__int64 a1)
{
  __int64 v1; // r12
  __int64 i; // rbx
  _QWORD *v3; // rdi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 240);
  for ( i = v1 + 16LL * *(unsigned int *)(a1 + 248); v1 != i; result = sub_B43D60(v3) )
  {
    v3 = *(_QWORD **)(i - 16);
    i -= 16;
  }
  return result;
}
