// Function: sub_16F2420
// Address: 0x16f2420
//
__int64 *__fastcall sub_16F2420(__int64 *a1, unsigned __int8 *a2, unsigned __int64 a3)
{
  size_t v4; // r15
  char *v5; // rax
  unsigned __int64 v6; // r14
  char *v7; // rcx
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  unsigned __int8 *v13; // [rsp+0h] [rbp-70h] BYREF
  char *v14; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  char *v18; // [rsp+28h] [rbp-48h]
  char *v19; // [rsp+30h] [rbp-40h]

  if ( a3 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v17 = 0;
  v18 = 0;
  v4 = 4 * a3;
  v19 = 0;
  if ( a3 )
  {
    v5 = (char *)sub_22077B0(4 * a3);
    v6 = (unsigned __int64)&v5[v4];
    v17 = (unsigned __int64)v5;
    v7 = v5;
    v19 = &v5[v4];
    if ( v5 != &v5[v4] )
      v7 = (char *)memset(v5, 0, v4);
  }
  else
  {
    v7 = 0;
    v6 = 0;
  }
  v14 = v7;
  v18 = (char *)v6;
  v13 = a2;
  sub_16F0F70(&v13, (char *)&a2[a3], (unsigned __int64 *)&v14, v6, 1);
  v9 = (unsigned __int64)v18;
  v10 = (__int64)&v18[-v17];
  if ( &v14[-v17] > &v18[-v17] )
  {
    sub_C17A60((__int64)&v17, ((__int64)&v14[-v17] >> 2) - (v10 >> 2));
    v10 = (__int64)&v18[-v17];
  }
  else if ( &v14[-v17] < &v18[-v17] && v14 != v18 )
  {
    v18 = v14;
    v10 = (__int64)&v14[-v17];
  }
  *a1 = (__int64)(a1 + 2);
  sub_2240A50(a1, v10, 0, v9, v8);
  v16 = *a1;
  v11 = a1[1] + v16;
  v15 = v17;
  sub_16F0D40(&v15, (unsigned __int64)v18, &v16, v11, 0);
  sub_22410F0(a1, v16 - *a1, 0);
  if ( v17 )
    j_j___libc_free_0(v17, &v19[-v17]);
  return a1;
}
