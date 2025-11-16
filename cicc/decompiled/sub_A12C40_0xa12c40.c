// Function: sub_A12C40
// Address: 0xa12c40
//
__int64 __fastcall sub_A12C40(__int64 *a1, unsigned int a2)
{
  __m128i *v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 *v11; // r12
  unsigned __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14[4]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v15[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v16; // [rsp+40h] [rbp-70h] BYREF
  __int64 v17; // [rsp+48h] [rbp-68h]
  __int64 v18; // [rsp+50h] [rbp-60h]
  __int64 v19; // [rsp+58h] [rbp-58h]
  __int64 v20; // [rsp+60h] [rbp-50h]
  unsigned __int64 v21; // [rsp+68h] [rbp-48h]
  __int64 v22; // [rsp+70h] [rbp-40h]
  __int64 v23; // [rsp+78h] [rbp-38h]
  __int64 v24; // [rsp+80h] [rbp-30h]
  __int64 v25; // [rsp+88h] [rbp-28h]

  v2 = (__m128i *)*a1;
  v3 = (__int64)(*(_QWORD *)(*a1 + 720) - *(_QWORD *)(*a1 + 712)) >> 4;
  if ( a2 < v3 )
    return sub_A08720(*a1, a2);
  if ( a2 < v2->m128i_i32[2] )
  {
    v4 = *(_QWORD *)(v2->m128i_i64[0] + 8LL * a2);
    if ( v4 )
      return v4;
  }
  if ( a2 < ((v2[46].m128i_i64[1] - v2[46].m128i_i64[0]) >> 3) + v3 )
  {
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    sub_A04120(&v16, 0);
    sub_A0FFA0(v2, a2, (__int64)&v16, v6);
    v4 = 0;
    sub_A10370(v2, (__int64)&v16, v7, v8, v9);
    if ( a2 < v2->m128i_i32[2] )
      v4 = *(_QWORD *)(v2->m128i_i64[0] + 8LL * a2);
    v14[0] = v22;
    v14[1] = v23;
    v14[2] = v24;
    v14[3] = v25;
    v15[0] = v18;
    v15[1] = v19;
    v15[2] = v20;
    v15[3] = v21;
    sub_A01C60(v15, v14);
    v10 = v16;
    if ( v16 )
    {
      v11 = (__int64 *)v21;
      v12 = v25 + 8;
      if ( v25 + 8 > v21 )
      {
        do
        {
          v13 = *v11++;
          j_j___libc_free_0(v13, 512);
        }
        while ( v12 > (unsigned __int64)v11 );
        v10 = v16;
      }
      j_j___libc_free_0(v10, 8 * v17);
    }
    return v4;
  }
  return sub_A07560(*a1, a2);
}
