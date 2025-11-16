// Function: sub_18C6E20
// Address: 0x18c6e20
//
__int64 __fastcall sub_18C6E20(__int64 a1, __int64 *a2)
{
  __int64 v2; // r8
  __int64 result; // rax
  __int64 v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v4[0] = *a2;
  *(_QWORD *)(a1 + 160) = sub_161BE60(v4, 0xFFFFFu, 1u);
  v2 = sub_16328F0((__int64)a2, "Cross-DSO CFI", 0xDu);
  result = 0;
  if ( v2 )
  {
    sub_18C5FA0(a1, a2);
    return 1;
  }
  return result;
}
