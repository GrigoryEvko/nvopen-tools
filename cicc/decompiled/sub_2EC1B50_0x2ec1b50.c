// Function: sub_2EC1B50
// Address: 0x2ec1b50
//
void __fastcall sub_2EC1B50(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD **v2; // rdx
  __int64 v3; // r12
  const void *v4; // rsi
  size_t v5; // rdx
  const __m128i *(__fastcall *v6)(__int64, const void *, size_t); // rax
  unsigned int v7; // eax
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // rcx
  const __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  _QWORD **v16; // rbx

  v1 = (_QWORD *)qword_5021050[0];
  if ( qword_5021050[0] )
  {
    if ( a1 == (_QWORD *)qword_5021050[0] )
    {
      v16 = (_QWORD **)qword_5021050;
LABEL_8:
      v3 = qword_5021050[2];
      if ( qword_5021050[2] )
      {
        v4 = (const void *)a1[1];
        v5 = a1[2];
        v6 = *(const __m128i *(__fastcall **)(__int64, const void *, size_t))(*(_QWORD *)qword_5021050[2] + 32LL);
        if ( v6 == sub_2EC1AA0 )
        {
          v7 = sub_C55C90(qword_5021050[2] + 8LL, v4, v5);
          v8 = *(_DWORD *)(v3 + 32);
          v9 = 56 * (v7 + 1LL);
          v10 = 56LL * v8 - v9;
          v11 = (const __m128i *)(v9 + *(_QWORD *)(v3 + 24));
          v12 = 0x6DB6DB6DB6DB6DB7LL * (v10 >> 3);
          if ( v10 > 0 )
          {
            do
            {
              v13 = v11[2].m128i_i64[1];
              v14 = _mm_loadu_si128(v11);
              v11 = (const __m128i *)((char *)v11 + 56);
              v15 = _mm_loadu_si128((const __m128i *)((char *)v11 - 40));
              v11[-5].m128i_i64[1] = v13;
              LOBYTE(v13) = v11[-1].m128i_i8[8];
              v11[-7] = v14;
              v11[-6] = v15;
              v11[-4].m128i_i8[0] = v13;
              --v12;
            }
            while ( v12 );
            v8 = *(_DWORD *)(v3 + 32);
          }
          *(_DWORD *)(v3 + 32) = v8 - 1;
        }
        else
        {
          v6(qword_5021050[2], v4, v5);
        }
      }
      *v16 = (_QWORD *)**v16;
    }
    else
    {
      while ( 1 )
      {
        v2 = (_QWORD **)v1;
        v1 = (_QWORD *)*v1;
        if ( !v1 )
          break;
        if ( a1 == v1 )
        {
          v16 = v2;
          goto LABEL_8;
        }
      }
    }
  }
}
