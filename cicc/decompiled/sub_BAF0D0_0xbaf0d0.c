// Function: sub_BAF0D0
// Address: 0xbaf0d0
//
_QWORD *__fastcall sub_BAF0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rdi
  _QWORD *result; // rax
  _BYTE *v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v18; // [rsp+30h] [rbp-30h] BYREF

  sub_CA0F50(&v14, a2);
  if ( v15 == 0x3FFFFFFFFFFFFFFFLL || v15 == 4611686018427387902LL )
    goto LABEL_13;
  sub_2241490(&v14, "=\"", 2, v7);
  sub_CA0F50(v17, a3);
  sub_2241490(&v14, v17[0], v17[1], v8);
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0], v18 + 1);
  if ( v15 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_13:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v14, "\"", 1, v9);
  v12 = *(__int64 **)(a1 + 8);
  if ( v12 == *(__int64 **)(a1 + 16) )
  {
    sub_8FD760((__m128i **)a1, *(const __m128i **)(a1 + 8), (__int64)&v14);
  }
  else
  {
    if ( v12 )
    {
      *v12 = (__int64)(v12 + 2);
      sub_BABE50(v12, v14, (__int64)&v14[v15]);
      v12 = *(__int64 **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v12 + 4;
  }
  sub_BAC200(a1, a4, v10, v11);
  result = v16;
  if ( v14 != (_BYTE *)v16 )
    return (_QWORD *)j_j___libc_free_0(v14, v16[0] + 1LL);
  return result;
}
