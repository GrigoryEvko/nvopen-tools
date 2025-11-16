// Function: sub_1BD47F0
// Address: 0x1bd47f0
//
__int64 *__fastcall sub_1BD47F0(
        __int64 ***a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 *v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]
  __int64 v20; // [rsp+28h] [rbp-38h]
  __int64 v21; // [rsp+30h] [rbp-30h]

  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v9 = sub_1BD33F0(a1, (__int64)&v15, a2, a3, a4, a5, a6, a7, a8, a9);
  v10 = v20;
  v11 = v19;
  v12 = v9;
  if ( v20 != v19 )
  {
    do
    {
      v13 = *(_QWORD *)(v11 + 8);
      if ( v13 != v11 + 24 )
        _libc_free(v13);
      v11 += 40;
    }
    while ( v10 != v11 );
    v11 = v19;
  }
  if ( v11 )
    j_j___libc_free_0(v11, v21 - v11);
  j___libc_free_0(v16);
  return v12;
}
