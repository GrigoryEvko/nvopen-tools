// Function: sub_2157750
// Address: 0x2157750
//
void __fastcall sub_2157750(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rdx
  unsigned __int64 *v5; // r9
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r8
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // r14
  __m128i si128; // xmm0
  _BYTE *v19; // rsi
  __m128i *v20; // rdx
  unsigned __int64 v21; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 *v22; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(unsigned __int64 **)(a1 + 864);
  v21 = a2;
  if ( v4 )
  {
    v5 = (unsigned __int64 *)(a1 + 856);
    v8 = v4;
    v9 = (unsigned __int64 *)(a1 + 856);
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v8[4] >= a2 )
          break;
        v8 = (unsigned __int64 *)v8[3];
        if ( !v11 )
          goto LABEL_6;
      }
      v9 = v8;
      v8 = (unsigned __int64 *)v8[2];
    }
    while ( v10 );
LABEL_6:
    if ( v5 != v9 )
    {
      v12 = v5;
      if ( v9[4] <= a2 )
      {
        do
        {
          while ( 1 )
          {
            v13 = v4[2];
            v14 = v4[3];
            if ( v4[4] >= a2 )
              break;
            v4 = (unsigned __int64 *)v4[3];
            if ( !v14 )
              goto LABEL_12;
          }
          v12 = v4;
          v4 = (unsigned __int64 *)v4[2];
        }
        while ( v13 );
LABEL_12:
        if ( v5 == v12 || v12[4] > a2 )
        {
          v22 = &v21;
          v12 = sub_21562D0((_QWORD *)(a1 + 848), v12, &v22);
        }
        v15 = (__int64)(v12[6] - v12[5]) >> 3;
        if ( (_DWORD)v15 )
        {
          v16 = 0;
          v17 = 8LL * (unsigned int)v15;
          do
          {
            v20 = *(__m128i **)(a3 + 24);
            if ( *(_QWORD *)(a3 + 16) - (_QWORD)v20 > 0x15u )
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_4327150);
              v20[1].m128i_i32[0] = 1701601889;
              v20[1].m128i_i16[2] = 2314;
              *v20 = si128;
              *(_QWORD *)(a3 + 24) += 22LL;
            }
            else
            {
              sub_16E7EE0(a3, "\t// demoted variable\n\t", 0x16u);
            }
            v19 = *(_BYTE **)(v12[5] + v16);
            v16 += 8;
            sub_2156420(a1, v19, a3, 1);
          }
          while ( v17 != v16 );
        }
      }
    }
  }
}
