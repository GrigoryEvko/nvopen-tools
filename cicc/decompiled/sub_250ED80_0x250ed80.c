// Function: sub_250ED80
// Address: 0x250ed80
//
void __fastcall sub_250ED80(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rax
  const __m128i *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r8
  __m128i *v15; // rax
  const void *v16; // rsi
  char *v17; // r12
  _QWORD v18[2]; // [rsp-48h] [rbp-48h] BYREF
  int v19; // [rsp-38h] [rbp-38h]

  if ( a4 != 2 )
  {
    if ( *(_DWORD *)(a1 + 408) )
    {
      v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 48LL))(a2);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v6 + 24LL))(v6) )
      {
        v8 = *(unsigned int *)(a1 + 408);
        v9 = *(_QWORD *)(a1 + 400);
        v18[0] = a2;
        v10 = (const __m128i *)v18;
        v18[1] = a3;
        v11 = *(_QWORD *)(v9 + 8 * v8 - 8);
        v19 = a4;
        v12 = *(unsigned int *)(v11 + 8);
        v13 = *(_QWORD *)v11;
        v14 = v12 + 1;
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
        {
          v16 = (const void *)(v11 + 16);
          if ( v13 > (unsigned __int64)v18 || (unsigned __int64)v18 >= v13 + 24 * v12 )
          {
            sub_C8D5F0(v11, v16, v14, 0x18u, v14, v7);
            v13 = *(_QWORD *)v11;
            v12 = *(unsigned int *)(v11 + 8);
          }
          else
          {
            v17 = (char *)v18 - v13;
            sub_C8D5F0(v11, v16, v14, 0x18u, v14, v7);
            v13 = *(_QWORD *)v11;
            v12 = *(unsigned int *)(v11 + 8);
            v10 = (const __m128i *)&v17[*(_QWORD *)v11];
          }
        }
        v15 = (__m128i *)(v13 + 24 * v12);
        *v15 = _mm_loadu_si128(v10);
        v15[1].m128i_i64[0] = v10[1].m128i_i64[0];
        ++*(_DWORD *)(v11 + 8);
      }
    }
  }
}
