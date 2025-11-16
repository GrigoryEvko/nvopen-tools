// Function: sub_2EC61D0
// Address: 0x2ec61d0
//
__int64 __fastcall sub_2EC61D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  __int64 v4; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 120);
  for ( i = v2 + 16LL * *(unsigned int *)(a2 + 128); i != v2; result = sub_2EC6140(a1, a2, v4) )
  {
    v4 = v2;
    v2 += 16;
  }
  return result;
}
