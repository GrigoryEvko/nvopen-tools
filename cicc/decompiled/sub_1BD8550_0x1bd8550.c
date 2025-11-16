// Function: sub_1BD8550
// Address: 0x1bd8550
//
__int64 __fastcall sub_1BD8550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  int v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+20h] [rbp-30h]
  __int64 v15; // [rsp+28h] [rbp-28h]
  __int64 v16; // [rsp+30h] [rbp-20h]

  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  sub_1BD7CA0(a1, a2, a3, (__int64)&v10, a4, a5, a6);
  v6 = v15;
  v7 = v14;
  if ( v15 != v14 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 + 8);
      if ( v8 != v7 + 24 )
        _libc_free(v8);
      v7 += 40;
    }
    while ( v6 != v7 );
    v7 = v14;
  }
  if ( v7 )
    j_j___libc_free_0(v7, v16 - v7);
  return j___libc_free_0(v11);
}
