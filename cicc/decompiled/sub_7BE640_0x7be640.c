// Function: sub_7BE640
// Address: 0x7be640
//
__int64 *sub_7BE640()
{
  __int64 *v0; // rbx
  unsigned int *v1; // rsi
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i v11; // xmm1
  __m128i v12; // xmm3
  __m128i v13; // xmm0
  __int64 *v14; // r13
  __int64 *v15; // r14
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 *v20; // [rsp+8h] [rbp-38h] BYREF

  v20 = 0;
  if ( word_4F06418[0] != 1 )
    return 0;
  v0 = (__int64 *)&v20;
  do
  {
    while ( 1 )
    {
      v1 = (unsigned int *)qword_4D04A00;
      v2 = sub_87EBB0(13, qword_4D04A00);
      *v0 = v2;
      v0 = (__int64 *)(v2 + 8);
      sub_7B8B50(0xDu, v1, v3, v4, v5, v6);
      if ( word_4F06418[0] == 29 )
        break;
      if ( word_4F06418[0] != 1 )
        goto LABEL_8;
    }
    sub_7B8B50(0xDu, v1, v7, v8, v9, v10);
    if ( !sub_7BE5B0(1u, 0x28u, 0, 0) )
    {
      v11 = _mm_loadu_si128(xmmword_4F06660);
      v12 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v13 = _mm_loadu_si128(&xmmword_4F06660[3]);
      unk_4D04A10 = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)&qword_4D04A00 = v11;
      unk_4D04A11 |= 0x20u;
      xmmword_4D04A20 = v12;
      qword_4D04A08 = *(_QWORD *)dword_4F07508;
      unk_4D04A30 = v13;
    }
  }
  while ( word_4F06418[0] == 1 );
LABEL_8:
  v14 = v20;
  if ( v20 )
  {
    v15 = v20;
    do
    {
      while ( 1 )
      {
        v17 = *v15;
        v16 = *(_QWORD *)(*v15 + 16);
        if ( v16 > 5 )
          break;
LABEL_14:
        v15 = (__int64 *)v15[1];
        if ( !v15 )
          return v14;
      }
      v18 = 7;
      if ( v16 <= 7 )
        v18 = *(_QWORD *)(*v15 + 16);
      if ( memcmp(*(const void **)(v17 + 8), "import", v18) )
      {
        if ( v16 > 7 )
          v16 = 7;
        if ( !memcmp(*(const void **)(v17 + 8), "module", v16) )
          sub_6851C0(0xC68u, (_DWORD *)v15 + 12);
        goto LABEL_14;
      }
      sub_6851C0(0xC67u, (_DWORD *)v15 + 12);
      v15 = (__int64 *)v15[1];
    }
    while ( v15 );
  }
  return v14;
}
