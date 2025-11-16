// Function: sub_1494CF0
// Address: 0x1494cf0
//
__int64 __fastcall sub_1494CF0(
        __int64 a1,
        _QWORD *a2,
        _QWORD *a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        __m128i a7,
        __m128i a8,
        unsigned __int8 a9)
{
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v14; // [rsp+0h] [rbp-1E0h] BYREF
  __int64 v15; // [rsp+8h] [rbp-1D8h]
  _QWORD *v16; // [rsp+10h] [rbp-1D0h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-1C8h]
  _QWORD *v18; // [rsp+1B0h] [rbp-30h] BYREF
  unsigned __int8 v19; // [rsp+1B8h] [rbp-28h]
  unsigned __int8 v20; // [rsp+1B9h] [rbp-27h]

  v10 = &v16;
  v11 = &v16;
  v14 = 0;
  v15 = 1;
  do
  {
    *v11 = -4;
    v11 += 13;
  }
  while ( v11 != &v18 );
  v19 = a5;
  v20 = a9;
  v18 = a3;
  sub_1494B40(a1, a2, (__int64)&v14, a3, a4, a5, a7, a8, a6, a9);
  if ( (v15 & 1) == 0 )
  {
    v10 = v16;
    if ( !v17 )
      goto LABEL_14;
    v11 = &v16[13 * v17];
  }
  do
  {
    if ( *v10 != -4 && *v10 != -16 )
    {
      v12 = v10[6];
      if ( v12 != v10[5] )
        _libc_free(v12);
    }
    v10 += 13;
  }
  while ( v10 != v11 );
  if ( (v15 & 1) != 0 )
    return a1;
  v10 = v16;
LABEL_14:
  j___libc_free_0(v10);
  return a1;
}
