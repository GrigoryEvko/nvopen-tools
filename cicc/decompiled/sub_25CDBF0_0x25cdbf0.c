// Function: sub_25CDBF0
// Address: 0x25cdbf0
//
__int64 __fastcall sub_25CDBF0(__int64 a1, __int64 **a2, __int64 a3, __m128i a4)
{
  __int64 *v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned __int64 v9; // [rsp+8h] [rbp-48h] BYREF
  __int64 v10[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v11[6]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *a2;
  v5 = *(_BYTE **)a3;
  v6 = *v4;
  v7 = *(_QWORD *)(a3 + 8);
  v10[0] = (__int64)v11;
  sub_25CCF50(v10, v5, (__int64)&v5[v7]);
  sub_25CD9B0(&v9, (__int64)v10, v6, a4);
  if ( (_QWORD *)v10[0] != v11 )
    j_j___libc_free_0(v10[0]);
  *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
  *(_QWORD *)a1 = v9;
  return a1;
}
