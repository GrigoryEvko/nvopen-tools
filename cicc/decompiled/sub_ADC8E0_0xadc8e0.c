// Function: sub_ADC8E0
// Address: 0xadc8e0
//
__int64 __fastcall sub_ADC8E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  unsigned int v5; // r12d
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // [rsp+8h] [rbp-28h]
  __int64 v10; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-18h]

  v5 = a5;
  v10 = a4;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v11 = 64;
  if ( a3 )
    v7 = sub_B9B140(v6, a2, a3);
  result = sub_B046D0(v6, &v10, v5, v7, 0, 1);
  if ( v11 > 0x40 )
  {
    if ( v10 )
    {
      v9 = result;
      j_j___libc_free_0_0(v10);
      return v9;
    }
  }
  return result;
}
