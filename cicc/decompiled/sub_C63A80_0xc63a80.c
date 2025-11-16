// Function: sub_C63A80
// Address: 0xc63a80
//
__int64 __fastcall sub_C63A80(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rcx
  _QWORD *v7; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+8h] [rbp-38h]
  _QWORD v9[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 56) )
    return sub_CB6200(a2, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16));
  (*(void (__fastcall **)(_QWORD **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 48) + 32LL))(
    &v7,
    *(_QWORD *)(a1 + 48),
    *(unsigned int *)(a1 + 40));
  result = sub_CB6200(a2, v7, v8);
  if ( v7 != v9 )
    result = j_j___libc_free_0(v7, v9[0] + 1LL);
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 )
  {
    v7 = v9;
    v8 = 0;
    LOBYTE(v9[0]) = 0;
    sub_2240E30(&v7, v4 + 1);
    if ( v8 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(&v7, " ", 1, v5);
    sub_2241490(&v7, *(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 16), v6);
    result = sub_CB6200(a2, v7, v8);
    if ( v7 != v9 )
      return j_j___libc_free_0(v7, v9[0] + 1LL);
  }
  return result;
}
