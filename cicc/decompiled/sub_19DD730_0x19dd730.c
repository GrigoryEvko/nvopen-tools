// Function: sub_19DD730
// Address: 0x19dd730
//
__int64 __fastcall sub_19DD730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  bool v6; // zf
  _QWORD *v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // r12
  __int64 result; // rax
  __int64 *v12[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v13[4]; // [rsp+10h] [rbp-20h] BYREF

  v6 = *(_BYTE *)(a2 + 16) == 35;
  v7 = *(_QWORD **)(a1 + 24);
  v13[0] = a3;
  v13[1] = a4;
  v12[0] = v13;
  v12[1] = (__int64 *)0x200000002LL;
  if ( v6 )
  {
    v8 = sub_147DD40((__int64)v7, (__int64 *)v12, 0, 0, a5, a6);
    v9 = v12[0];
    v10 = (__int64)v8;
    if ( v12[0] == v13 )
      return v10;
LABEL_3:
    _libc_free((unsigned __int64)v9);
    return v10;
  }
  result = sub_147EE30(v7, v12, 0, 0, a5, a6);
  v9 = v12[0];
  v10 = result;
  if ( v12[0] != v13 )
    goto LABEL_3;
  return result;
}
