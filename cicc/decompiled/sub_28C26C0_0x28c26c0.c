// Function: sub_28C26C0
// Address: 0x28c26c0
//
__int64 __fastcall sub_28C26C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 *v4; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(a2 - 32);
  v4 = sub_DD8400(*(_QWORD *)(a1 + 24), a2);
  if ( sub_D968A0((__int64)v4) )
    return 0;
  result = sub_28C22C0(a1, v2, v3, a2);
  if ( !result )
    return sub_28C22C0(a1, v3, v2, a2);
  return result;
}
