// Function: sub_12E8D50
// Address: 0x12e8d50
//
__int64 __fastcall sub_12E8D50(__int64 *a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v4[6]; // [rsp+10h] [rbp-30h] BYREF

  v1 = *a1;
  v3[0] = (__int64)v4;
  sub_12D3F10(v3, *(_BYTE **)(v1 + 80), *(_QWORD *)(v1 + 80) + *(_QWORD *)(v1 + 88));
  result = sub_12E86C0(v1, *(_DWORD *)(v1 + 4624), v1 + 112, v3);
  if ( (_QWORD *)v3[0] != v4 )
    return j_j___libc_free_0(v3[0], v4[0] + 1LL);
  return result;
}
