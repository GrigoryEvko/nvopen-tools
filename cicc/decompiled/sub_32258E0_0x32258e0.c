// Function: sub_32258E0
// Address: 0x32258e0
//
void *__fastcall sub_32258E0(__int64 a1, unsigned __int64 a2, void **a3, unsigned int a4)
{
  unsigned int v4; // r15d
  __int64 v7; // rax
  char v8; // si
  char *v9; // rax
  unsigned int v10; // r14d
  char *v11; // rax
  char *v13; // rax
  __m128i **v14; // rbx
  __int64 i; // rbx
  __m128i **v16; // [rsp+10h] [rbp-A0h]
  __m128i v18; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v20[3]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v21; // [rsp+58h] [rbp-58h]
  char *v22; // [rsp+60h] [rbp-50h]
  __int64 v23; // [rsp+68h] [rbp-48h]
  __int64 v24; // [rsp+70h] [rbp-40h]

  v4 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v23 = 0x100000000LL;
  v20[1] = 2;
  v20[2] = 0;
  v20[0] = &unk_49DD288;
  v21 = 0;
  v22 = 0;
  v24 = v7;
  sub_CB5980((__int64)v20, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      ++v4;
      v8 = a2 & 0x7F;
      a2 >>= 7;
      if ( a2 || a4 > v4 )
        v8 |= 0x80u;
      v9 = v22;
      if ( (unsigned __int64)v22 >= v21 )
        break;
      ++v22;
      *v9 = v8;
      if ( !a2 )
        goto LABEL_7;
    }
    sub_CB5D20((__int64)v20, v8);
  }
  while ( a2 );
LABEL_7:
  if ( a4 > v4 )
  {
    v10 = a4 - 1;
    if ( v4 < v10 )
    {
      do
      {
        v13 = v22;
        if ( (unsigned __int64)v22 < v21 )
        {
          ++v22;
          *v13 = 0x80;
        }
        else
        {
          sub_CB5D20((__int64)v20, 128);
        }
        ++v4;
      }
      while ( v4 != v10 );
    }
    else
    {
      v10 = v4;
    }
    v11 = v22;
    if ( (unsigned __int64)v22 >= v21 )
    {
      sub_CB5D20((__int64)v20, 0);
    }
    else
    {
      ++v22;
      *v11 = 0;
    }
    v4 = v10 + 1;
  }
  if ( *(_BYTE *)(a1 + 24) )
  {
    v14 = *(__m128i ***)(a1 + 16);
    sub_CA0F50(v18.m128i_i64, a3);
    sub_3225850(v14, &v18);
    if ( (_QWORD *)v18.m128i_i64[0] != v19 )
      j_j___libc_free_0(v18.m128i_u64[0]);
    if ( v4 > 1 )
    {
      for ( i = 1; i != v4; ++i )
      {
        v16 = *(__m128i ***)(a1 + 16);
        v18.m128i_i64[0] = (__int64)v19;
        sub_3219430(v18.m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
        sub_3225850(v16, &v18);
        if ( (_QWORD *)v18.m128i_i64[0] != v19 )
          j_j___libc_free_0(v18.m128i_u64[0]);
      }
    }
  }
  v20[0] = &unk_49DD388;
  return sub_CB5840((__int64)v20);
}
