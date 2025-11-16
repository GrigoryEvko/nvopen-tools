// Function: sub_CEFFE0
// Address: 0xceffe0
//
__int64 *__fastcall sub_CEFFE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 *v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  v2 = *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  v6 = sub_AE2980(a2, *(_DWORD *)(v2 + 8) >> 8)[1];
  if ( v6 > 0x40 )
    sub_C43690((__int64)&v5, 0, 0);
  else
    v5 = 0;
  sub_B4DE60(a1, a2, (__int64)&v5);
  if ( v6 <= 0x40 )
    return v5;
  v3 = *v5;
  j_j___libc_free_0_0(v5);
  return (__int64 *)v3;
}
