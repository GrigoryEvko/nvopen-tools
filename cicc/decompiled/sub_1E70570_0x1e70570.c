// Function: sub_1E70570
// Address: 0x1e70570
//
__int64 __fastcall sub_1E70570(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  __int64 v4; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 32);
  for ( i = v2 + 16LL * *(unsigned int *)(a2 + 40); i != v2; result = sub_1E704F0(a1, a2, v4) )
  {
    v4 = v2;
    v2 += 16;
  }
  return result;
}
