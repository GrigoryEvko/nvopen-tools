// Function: sub_AD0760
// Address: 0xad0760
//
__int64 __fastcall sub_AD0760(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  __int64 v11; // rax
  int v12; // eax
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 *v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r9
  __int64 *v21; // r15
  unsigned int v22; // esi
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 **v25; // rdx
  __int64 *v26; // rax
  __int64 **v27; // r15
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdx
  int i; // [rsp+4h] [rbp-8Ch]
  int j; // [rsp+8h] [rbp-88h]
  __int64 **v34; // [rsp+18h] [rbp-78h] BYREF
  __m128i v35; // [rsp+20h] [rbp-70h] BYREF
  __int64 v36; // [rsp+30h] [rbp-60h]
  int v37; // [rsp+40h] [rbp-50h] BYREF
  __m128i v38; // [rsp+48h] [rbp-48h]
  __int64 v39; // [rsp+58h] [rbp-38h]

  v11 = a4[1];
  v35.m128i_i64[1] = (__int64)a2;
  v36 = a3;
  v35.m128i_i64[0] = v11;
  v37 = sub_AC5F60(a2, (__int64)&a2[a3]);
  v12 = sub_AC7AE0(v35.m128i_i64, &v37);
  v13 = _mm_loadu_si128(&v35);
  v14 = *(unsigned int *)(a1 + 24);
  v15 = *(_QWORD *)(a1 + 8);
  v37 = v12;
  v39 = v36;
  v38 = v13;
  if ( (_DWORD)v14 )
  {
    v16 = (v14 - 1) & v12;
    v17 = (__int64 *)(v15 + 8LL * v16);
    v18 = *v17;
    if ( *v17 != -4096 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v18 != -8192 && v38.m128i_i64[0] == *(_QWORD *)(v18 + 8) && v36 == 4 )
        {
          v19 = 0;
          while ( *(_QWORD *)(v38.m128i_i64[1] + 8 * v19) == *(_QWORD *)(v18 + 32 * v19 - 128) )
          {
            if ( ++v19 == 4 )
            {
              if ( v17 == (__int64 *)(v15 + 8 * v14) )
                goto LABEL_14;
              return *v17;
            }
          }
        }
        v16 = (v14 - 1) & (i + v16);
        v17 = (__int64 *)(v15 + 8LL * v16);
        v18 = *v17;
        if ( *v17 == -4096 )
          break;
      }
    }
  }
LABEL_14:
  v21 = a4 - 16;
  sub_AC7B60(a1, a4);
  if ( a7 == 1 )
  {
    sub_AC2B30((__int64)&a4[4 * a8 - 16], a6);
  }
  else
  {
    do
    {
      if ( a5 == *v21 )
        sub_AC2B30((__int64)v21, a6);
      v21 += 4;
    }
    while ( a4 != v21 );
  }
  v22 = *(_DWORD *)(a1 + 24);
  if ( !v22 )
  {
    ++*(_QWORD *)a1;
    v34 = 0;
LABEL_40:
    v22 *= 2;
    goto LABEL_41;
  }
  v23 = *(_QWORD *)(a1 + 8);
  v24 = (v22 - 1) & v37;
  v25 = (__int64 **)(v23 + 8LL * v24);
  v26 = *v25;
  if ( *v25 != (__int64 *)-4096LL )
  {
    v27 = 0;
    for ( j = 1; ; ++j )
    {
      if ( v26 == (__int64 *)-8192LL )
      {
        if ( !v27 )
          v27 = v25;
      }
      else if ( v38.m128i_i64[0] == v26[1] && v39 == 4 )
      {
        v30 = 0;
        while ( *(_QWORD *)(v38.m128i_i64[1] + 8 * v30) == v26[4 * v30 - 16] )
        {
          if ( ++v30 == 4 )
            return 0;
        }
      }
      v24 = (v22 - 1) & (j + v24);
      v25 = (__int64 **)(v23 + 8LL * v24);
      v26 = *v25;
      if ( *v25 == (__int64 *)-4096LL )
        break;
    }
    if ( v27 )
      v25 = v27;
  }
  v28 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v34 = v25;
  v29 = v28 + 1;
  if ( 4 * v29 >= 3 * v22 )
    goto LABEL_40;
  if ( v22 - *(_DWORD *)(a1 + 20) - v29 <= v22 >> 3 )
  {
LABEL_41:
    sub_AD00B0(a1, v22);
    sub_AC7F30(a1, (__int64)&v37, &v34);
    v25 = v34;
    v29 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v29;
  if ( *v25 != (__int64 *)-4096LL )
    --*(_DWORD *)(a1 + 20);
  *v25 = a4;
  return 0;
}
