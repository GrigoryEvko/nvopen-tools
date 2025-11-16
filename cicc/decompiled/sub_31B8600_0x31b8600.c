// Function: sub_31B8600
// Address: 0x31b8600
//
__m128i *__fastcall sub_31B8600(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v11; // r13
  __m128i v12; // xmm0
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r8
  __m128i v19; // xmm1
  int v21; // edx
  int v22; // r9d
  __m128i v23; // [rsp+0h] [rbp-70h] BYREF
  __int64 v24; // [rsp+10h] [rbp-60h]
  __m128i v25[5]; // [rsp+20h] [rbp-50h] BYREF

  v11 = a9;
  if ( (_QWORD)a7 == a9 )
    goto LABEL_12;
  while ( 1 )
  {
    v12 = _mm_loadu_si128((const __m128i *)&a7);
    v24 = a8;
    v23 = v12;
    sub_318E780(v25, &v23);
    v13 = sub_318E5D0((__int64)v25);
    if ( sub_318B630(v13) )
    {
      if ( v13 )
      {
        v14 = *(unsigned int *)(a2 + 24);
        v15 = *(_QWORD *)(a2 + 8);
        if ( (_DWORD)v14 )
        {
          v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v13 != *v17 )
          {
            v21 = 1;
            while ( v18 != -4096 )
            {
              v22 = v21 + 1;
              v16 = (v14 - 1) & (v21 + v16);
              v17 = (__int64 *)(v15 + 16LL * v16);
              v18 = *v17;
              if ( v13 == *v17 )
                goto LABEL_6;
              v21 = v22;
            }
            goto LABEL_11;
          }
LABEL_6:
          if ( v17 != (__int64 *)(v15 + 16 * v14) && v17[1] )
            break;
        }
      }
    }
LABEL_11:
    sub_318E7A0((__int64)&a7);
    if ( (_QWORD)a7 == v11 )
    {
LABEL_12:
      if ( *((_QWORD *)&a7 + 1) == a10 )
        break;
    }
  }
  v19 = _mm_loadu_si128((const __m128i *)&a7);
  a1[1].m128i_i64[0] = a8;
  *a1 = v19;
  return a1;
}
