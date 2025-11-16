// Function: sub_1CAD270
// Address: 0x1cad270
//
const char *__fastcall sub_1CAD270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  bool v8; // cc
  __m128i si128; // xmm0
  __m128i *v10; // rax
  const char *result; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r14
  const char *v14; // r15
  int v15; // r9d
  unsigned __int64 v16; // r12
  const void *v17; // rsi
  char v18; // r13
  char v19; // cl
  unsigned __int8 v20; // [rsp+7h] [rbp-39h]

  v6 = 0;
  v8 = *(_DWORD *)(a2 + 12) <= 0x1Au;
  *(_DWORD *)(a2 + 8) = 0;
  if ( v8 )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0x1Bu, 1, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  si128 = _mm_load_si128((const __m128i *)&xmmword_42DFE40);
  v10 = (__m128i *)(*(_QWORD *)a2 + v6);
  qmemcpy(&v10[1], "4bit.index.", 11);
  *v10 = si128;
  *(_DWORD *)(a2 + 8) += 27;
  result = sub_1649960(a1);
  v13 = v12;
  if ( v12 )
  {
    v14 = result;
    if ( *result != 1 || (v14 = result + 1, v13 = v12 - 1, v12 != 1) )
    {
      v15 = 0;
      v16 = 0;
      v17 = (const void *)(a2 + 16);
      while ( 1 )
      {
        v18 = v14[v16];
        if ( v18 == 91 )
        {
          while ( 1 )
          {
            result = (const char *)*(unsigned int *)(a2 + 8);
            if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
              break;
LABEL_12:
            ++v16;
            result[*(_QWORD *)a2] = 46;
            ++*(_DWORD *)(a2 + 8);
            if ( v13 <= v16 )
              return result;
            v18 = v14[v16];
            if ( v18 != 91 )
              goto LABEL_14;
          }
LABEL_17:
          sub_16CD150(a2, v17, 0, 1, 67111681, v15);
          result = (const char *)*(unsigned int *)(a2 + 8);
          goto LABEL_12;
        }
        if ( !(_BYTE)v15 )
          goto LABEL_8;
LABEL_14:
        v19 = v18 - 32;
        if ( (unsigned __int8)(v18 - 32) <= 0x3Du )
          break;
        result = (const char *)*(unsigned int *)(a2 + 8);
        v15 = 1;
        if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
        {
LABEL_21:
          v20 = v15;
          sub_16CD150(a2, v17, 0, 1, 67111681, v15);
          result = (const char *)*(unsigned int *)(a2 + 8);
          v15 = v20;
        }
LABEL_9:
        ++v16;
        result[*(_QWORD *)a2] = v18;
        ++*(_DWORD *)(a2 + 8);
        if ( v13 <= v16 )
          return result;
      }
      v15 = ((0x2000000004000B01uLL >> v19) & 1) == 0;
      if ( ((0x2000000004000B01uLL >> v19) & 1) != 0 )
      {
        result = (const char *)*(unsigned int *)(a2 + 8);
        if ( (unsigned int)result < *(_DWORD *)(a2 + 12) )
          goto LABEL_12;
        goto LABEL_17;
      }
LABEL_8:
      result = (const char *)*(unsigned int *)(a2 + 8);
      if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
        goto LABEL_21;
      goto LABEL_9;
    }
  }
  return result;
}
