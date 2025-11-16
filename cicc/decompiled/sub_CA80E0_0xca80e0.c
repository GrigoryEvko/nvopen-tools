// Function: sub_CA80E0
// Address: 0xca80e0
//
void __fastcall sub_CA80E0(__int64 a1, __int64 a2, int a3, char a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  __m128i *v13; // rax
  __int64 v14; // rdi
  const void *v15; // rsi
  char *v16; // r12
  __int64 v17; // [rsp-38h] [rbp-38h] BYREF
  int v18; // [rsp-30h] [rbp-30h]
  __int64 v19; // [rsp-2Ch] [rbp-2Ch]
  char v20; // [rsp-24h] [rbp-24h]
  __int16 v21; // [rsp-23h] [rbp-23h]
  char v22; // [rsp-21h] [rbp-21h]

  if ( *(_BYTE *)(a1 + 73) )
  {
    v6 = (const __m128i *)&v17;
    v21 = 0;
    v8 = *(_QWORD *)(a1 + 64);
    v20 = a4;
    v9 = *(unsigned int *)(a1 + 236);
    v19 = v8;
    v10 = *(unsigned int *)(a1 + 232);
    v18 = a3;
    v11 = *(_QWORD *)(a1 + 224);
    v12 = v10 + 1;
    v22 = 0;
    v17 = a2;
    if ( v10 + 1 > v9 )
    {
      v14 = a1 + 224;
      v15 = (const void *)(a1 + 240);
      if ( v11 > (unsigned __int64)&v17 || (unsigned __int64)&v17 >= v11 + 24 * v10 )
      {
        sub_C8D5F0(v14, v15, v12, 0x18u, v12, a6);
        v11 = *(_QWORD *)(a1 + 224);
        v10 = *(unsigned int *)(a1 + 232);
      }
      else
      {
        v16 = (char *)&v17 - v11;
        sub_C8D5F0(v14, v15, v12, 0x18u, v12, a6);
        v11 = *(_QWORD *)(a1 + 224);
        v10 = *(unsigned int *)(a1 + 232);
        v6 = (const __m128i *)&v16[v11];
      }
    }
    v13 = (__m128i *)(v11 + 24 * v10);
    *v13 = _mm_loadu_si128(v6);
    v13[1].m128i_i64[0] = v6[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 232);
  }
}
