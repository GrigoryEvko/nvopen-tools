// Function: sub_A1BEE0
// Address: 0xa1bee0
//
__int64 __fastcall sub_A1BEE0(__int64 a1)
{
  __int64 v2; // r8
  size_t v3; // rsi
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]
  __int64 v8; // [rsp+10h] [rbp-20h]

  v6 = 0;
  v7 = 0;
  v8 = 0;
  sub_C0D2A0(a1 + 8);
  v2 = 0;
  v3 = *(_QWORD *)(a1 + 40);
  if ( v3 )
  {
    sub_A1A250(&v6, v3);
    v2 = v6;
  }
  sub_C0BFF0(a1 + 8, v2);
  result = sub_A1B8B0((__int64 *)a1, 0x17u, 1u, v6, v7 - v6);
  v5 = v6;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v5 )
    return j_j___libc_free_0(v5, v8 - v5);
  return result;
}
