// Function: sub_3225C30
// Address: 0x3225c30
//
void *__fastcall sub_3225C30(__int64 a1, __int64 a2, void **a3)
{
  unsigned int v3; // r15d
  __int64 v5; // rax
  char v6; // r13
  char v7; // si
  char *v8; // rax
  char v9; // al
  __m128i **v11; // rbx
  __int64 i; // rbx
  __m128i **v13; // [rsp+10h] [rbp-A0h]
  __m128i v15; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v16[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v17[3]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v18; // [rsp+58h] [rbp-58h]
  char *v19; // [rsp+60h] [rbp-50h]
  __int64 v20; // [rsp+68h] [rbp-48h]
  __int64 v21; // [rsp+70h] [rbp-40h]

  v3 = 0;
  v5 = *(_QWORD *)(a1 + 8);
  v20 = 0x100000000LL;
  v17[1] = 2;
  v17[2] = 0;
  v17[0] = &unk_49DD288;
  v18 = 0;
  v19 = 0;
  v21 = v5;
  sub_CB5980((__int64)v17, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      v9 = a2;
      ++v3;
      v7 = a2 & 0x7F;
      a2 >>= 7;
      if ( !a2 )
      {
        v6 = 0;
        if ( (v9 & 0x40) == 0 )
          goto LABEL_4;
        goto LABEL_3;
      }
      if ( a2 == -1 )
      {
        v6 = 0;
        if ( (v9 & 0x40) != 0 )
          break;
      }
LABEL_3:
      v7 |= 0x80u;
      v6 = 1;
LABEL_4:
      v8 = v19;
      if ( (unsigned __int64)v19 >= v18 )
        goto LABEL_10;
LABEL_5:
      v19 = v8 + 1;
      *v8 = v7;
      if ( !v6 )
        goto LABEL_11;
    }
    v8 = v19;
    if ( (unsigned __int64)v19 < v18 )
      goto LABEL_5;
LABEL_10:
    sub_CB5D20((__int64)v17, v7);
  }
  while ( v6 );
LABEL_11:
  if ( *(_BYTE *)(a1 + 24) )
  {
    v11 = *(__m128i ***)(a1 + 16);
    sub_CA0F50(v15.m128i_i64, a3);
    sub_3225850(v11, &v15);
    if ( (_QWORD *)v15.m128i_i64[0] != v16 )
      j_j___libc_free_0(v15.m128i_u64[0]);
    if ( v3 > 1 )
    {
      for ( i = 1; i != v3; ++i )
      {
        v13 = *(__m128i ***)(a1 + 16);
        v15.m128i_i64[0] = (__int64)v16;
        sub_3219430(v15.m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
        sub_3225850(v13, &v15);
        if ( (_QWORD *)v15.m128i_i64[0] != v16 )
          j_j___libc_free_0(v15.m128i_u64[0]);
      }
    }
  }
  v17[0] = &unk_49DD388;
  return sub_CB5840((__int64)v17);
}
