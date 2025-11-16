// Function: sub_90A4E0
// Address: 0x90a4e0
//
__int64 __fastcall sub_90A4E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]

  v7 = 0;
  v8 = 0;
  v9 = 0;
  sub_93AE30(a1, a3, a2, &v7);
  result = sub_A7B050(*(_QWORD *)(a1 + 344), v7, (v8 - v7) >> 3);
  v6 = v7;
  *(_QWORD *)(a4 + 120) = result;
  if ( v6 )
    return j_j___libc_free_0(v6, v9 - v6);
  return result;
}
