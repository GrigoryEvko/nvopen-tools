// Function: sub_3225B70
// Address: 0x3225b70
//
void __fastcall sub_3225B70(__int64 a1, char a2, void **a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  __int64 v8; // rax
  __m128i **v9; // r13
  __m128i v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF

  v7 = *(_QWORD **)(a1 + 8);
  v8 = v7[1];
  if ( (unsigned __int64)(v8 + 1) > v7[2] )
  {
    sub_C8D290(*(_QWORD *)(a1 + 8), v7 + 3, v8 + 1, 1u, a5, a6);
    v8 = v7[1];
  }
  *(_BYTE *)(*v7 + v8) = a2;
  ++v7[1];
  if ( *(_BYTE *)(a1 + 24) )
  {
    v9 = *(__m128i ***)(a1 + 16);
    sub_CA0F50(v10.m128i_i64, a3);
    sub_3225850(v9, &v10);
    if ( (__int64 *)v10.m128i_i64[0] != &v11 )
      j_j___libc_free_0(v10.m128i_u64[0]);
  }
}
