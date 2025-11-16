// Function: sub_1904910
// Address: 0x1904910
//
__int64 __fastcall sub_1904910(__int64 a1)
{
  __int64 v1; // r12
  __int64 i; // rbx
  _QWORD *v3; // rdi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 240);
  for ( i = *(_QWORD *)(a1 + 248); v1 != i; result = sub_15F20C0(v3) )
  {
    v3 = *(_QWORD **)(i - 16);
    i -= 16;
  }
  return result;
}
