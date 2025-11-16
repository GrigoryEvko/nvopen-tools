// Function: sub_152B5E0
// Address: 0x152b5e0
//
__int64 __fastcall sub_152B5E0(__int64 a1)
{
  unsigned __int8 *v2; // r8
  size_t v3; // rsi
  __int64 result; // rax
  unsigned __int8 *v5; // rdi
  unsigned __int8 *v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]
  __int64 v8; // [rsp+10h] [rbp-20h]

  v6 = 0;
  v7 = 0;
  v8 = 0;
  sub_16805A0(a1 + 16);
  v2 = 0;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 )
  {
    sub_A1A250((__int64 *)&v6, v3);
    v2 = v6;
  }
  sub_167FAF0(a1 + 16, v2);
  result = sub_152B430(a1, 0x17u, 1u, v6, v7 - (_QWORD)v6);
  v5 = v6;
  *(_BYTE *)(a1 + 176) = 1;
  if ( v5 )
    return j_j___libc_free_0(v5, v8 - (_QWORD)v5);
  return result;
}
