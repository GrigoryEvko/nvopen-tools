// Function: sub_1E40520
// Address: 0x1e40520
//
void __fastcall sub_1E40520(_QWORD *a1)
{
  __int64 v1; // r14
  unsigned __int64 v2; // rax
  _QWORD *v3; // r9
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // r8
  unsigned __int64 v7; // r14
  const __m128i *v8; // rbx
  const __m128i *i; // r13
  __m128i *v10; // rax
  unsigned __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned int v14; // r15d
  unsigned int v15; // r14d
  __int64 v16; // [rsp+8h] [rbp-138h]
  _QWORD *v17; // [rsp+10h] [rbp-130h]
  _BYTE *v18; // [rsp+18h] [rbp-128h]
  unsigned __int64 v19; // [rsp+20h] [rbp-120h] BYREF
  __m128i v20; // [rsp+28h] [rbp-118h]
  _BYTE *v21; // [rsp+40h] [rbp-100h] BYREF
  __int64 v22; // [rsp+48h] [rbp-F8h]
  _BYTE v23[240]; // [rsp+50h] [rbp-F0h] BYREF

  v1 = *a1;
  v21 = v23;
  v22 = 0x800000000LL;
  v2 = 0xF0F0F0F0F0F0F0F1LL * ((a1[1] - v1) >> 4);
  if ( (_DWORD)v2 )
  {
    v3 = a1;
    v4 = 0;
    v5 = 0;
    v6 = 272LL * (unsigned int)(v2 - 1);
    while ( 1 )
    {
      v7 = v4 + v1;
      v8 = *(const __m128i **)(v7 + 32);
      for ( i = &v8[*(unsigned int *)(v7 + 40)]; i != v8; LODWORD(v22) = v22 + 1 )
      {
        while ( ((v8->m128i_i64[0] >> 1) & 3) != 1 )
        {
          if ( i == ++v8 )
            goto LABEL_10;
        }
        v19 = v7;
        v20 = _mm_loadu_si128(v8);
        if ( HIDWORD(v22) <= (unsigned int)v5 )
        {
          v16 = v6;
          v17 = v3;
          sub_16CD150((__int64)&v21, v23, 0, 24, v6, (int)v3);
          v5 = (unsigned int)v22;
          v6 = v16;
          v3 = v17;
        }
        ++v8;
        v10 = (__m128i *)&v21[24 * v5];
        *v10 = _mm_loadu_si128((const __m128i *)&v19);
        v10[1].m128i_i64[0] = v20.m128i_i64[1];
        v5 = (unsigned int)(v22 + 1);
      }
LABEL_10:
      if ( v6 == v4 )
        break;
      v1 = *v3;
      v4 += 272;
    }
    v11 = (unsigned __int64)v21;
    v18 = &v21[24 * v5];
    if ( v18 != v21 )
    {
      do
      {
        v12 = *(_QWORD *)v11;
        v13 = *(_QWORD *)(v11 + 8);
        v11 += 24LL;
        v14 = *(_DWORD *)(v11 - 8);
        v15 = *(_DWORD *)(v11 - 4);
        sub_1F01C30(v12);
        v19 = v12 & 0xFFFFFFFFFFFFFFF9LL | 2;
        v20.m128i_i64[0] = __PAIR64__(v15, v14);
        sub_1F01A00(v13 & 0xFFFFFFFFFFFFFFF8LL, &v19, 1);
      }
      while ( (_BYTE *)v11 != v18 );
      v11 = (unsigned __int64)v21;
    }
    if ( (_BYTE *)v11 != v23 )
      _libc_free(v11);
  }
}
