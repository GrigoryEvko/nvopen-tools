// Function: sub_2CF5D20
// Address: 0x2cf5d20
//
const char *__fastcall sub_2CF5D20(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  bool v8; // cc
  __m128i si128; // xmm0
  __m128i *v10; // rax
  const char *result; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // r8
  unsigned __int64 v14; // r15
  bool v15; // r14
  unsigned __int64 v16; // r12
  const void *v17; // rsi
  char v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rdx
  char v21; // cl
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v23; // [rsp+0h] [rbp-40h]

  v6 = 0;
  v8 = a2[2] <= 0x1Au;
  a2[1] = 0;
  if ( v8 )
  {
    sub_C8D290((__int64)a2, a2 + 3, 27, 1u, a5, a6);
    v6 = a2[1];
  }
  si128 = _mm_load_si128((const __m128i *)&xmmword_42DFE40);
  v10 = (__m128i *)(*a2 + v6);
  qmemcpy(&v10[1], "4bit.index.", 11);
  *v10 = si128;
  a2[1] += 27LL;
  result = sub_BD5D20(a1);
  v13 = (__int64)result;
  v14 = v12;
  if ( v12 )
  {
    if ( *result != 1 || (v13 = (__int64)(result + 1), v14 = v12 - 1, v12 != 1) )
    {
      v15 = 0;
      v16 = 0;
      v17 = a2 + 3;
      while ( 1 )
      {
        v18 = *(_BYTE *)(v13 + v16);
        if ( v18 == 91 )
        {
          while ( 1 )
          {
            result = (const char *)a2[1];
            v20 = (__int64)(result + 1);
            if ( (unsigned __int64)(result + 1) > a2[2] )
              break;
LABEL_12:
            ++v16;
            result[*a2] = 46;
            ++a2[1];
            if ( v14 <= v16 )
              return result;
            v18 = *(_BYTE *)(v13 + v16);
            if ( v18 != 91 )
              goto LABEL_14;
          }
LABEL_17:
          v22 = v13;
          sub_C8D290((__int64)a2, v17, v20, 1u, v13, 0x2000000004000B01LL);
          result = (const char *)a2[1];
          v13 = v22;
          goto LABEL_12;
        }
        if ( !v15 )
          goto LABEL_8;
LABEL_14:
        v21 = v18 - 32;
        if ( (unsigned __int8)(v18 - 32) <= 0x3Du )
          break;
        result = (const char *)a2[1];
        v15 = 1;
        v19 = (__int64)(result + 1);
        if ( (unsigned __int64)(result + 1) > a2[2] )
        {
LABEL_21:
          v23 = v13;
          sub_C8D290((__int64)a2, v17, v19, 1u, v13, 0x2000000004000B01LL);
          result = (const char *)a2[1];
          v13 = v23;
        }
LABEL_9:
        ++v16;
        result[*a2] = v18;
        ++a2[1];
        if ( v14 <= v16 )
          return result;
      }
      v15 = ((0x2000000004000B01uLL >> v21) & 1) == 0;
      if ( ((0x2000000004000B01uLL >> v21) & 1) != 0 )
      {
        result = (const char *)a2[1];
        v20 = (__int64)(result + 1);
        if ( (unsigned __int64)(result + 1) <= a2[2] )
          goto LABEL_12;
        goto LABEL_17;
      }
LABEL_8:
      result = (const char *)a2[1];
      v19 = (__int64)(result + 1);
      if ( (unsigned __int64)(result + 1) > a2[2] )
        goto LABEL_21;
      goto LABEL_9;
    }
  }
  return result;
}
