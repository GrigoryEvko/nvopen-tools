// Function: sub_39F6770
// Address: 0x39f6770
//
unsigned __int64 __fastcall sub_39F6770(const __m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r11
  __int64 v7; // r11
  __int64 v8; // rsi
  int v9; // ecx
  char *v10; // rdi
  char v11; // dl
  unsigned __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rbx
  char *v15; // r8
  __int8 *v16; // rbp
  __int64 *v17; // r13
  __int8 *v18; // r14
  char *v19; // rdi
  __int64 v20; // rsi
  int v21; // ecx
  char v22; // dl
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 result; // rax
  char *v26; // rdi
  __int64 v27; // rsi
  int v28; // ecx
  char v29; // dl
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  const __m128i *v32; // [rsp+8h] [rbp-160h]
  char *v33; // [rsp+18h] [rbp-150h]
  char *v34; // [rsp+18h] [rbp-150h]
  __int64 v36; // [rsp+38h] [rbp-130h] BYREF
  _OWORD v37[3]; // [rsp+40h] [rbp-128h] BYREF
  __m128i v38; // [rsp+70h] [rbp-F8h]
  __m128i v39; // [rsp+80h] [rbp-E8h]
  __m128i v40; // [rsp+90h] [rbp-D8h]
  __m128i v41; // [rsp+A0h] [rbp-C8h]
  __m128i v42; // [rsp+B0h] [rbp-B8h]
  __m128i v43; // [rsp+C0h] [rbp-A8h]
  __m128i v44; // [rsp+D0h] [rbp-98h]
  __m128i v45; // [rsp+E0h] [rbp-88h]
  __m128i v46; // [rsp+F0h] [rbp-78h]
  __m128i v47; // [rsp+100h] [rbp-68h]
  _OWORD v48[5]; // [rsp+110h] [rbp-58h]

  v37[0] = _mm_loadu_si128(a1);
  v37[1] = _mm_loadu_si128(a1 + 1);
  v37[2] = _mm_loadu_si128(a1 + 2);
  v38 = _mm_loadu_si128(a1 + 3);
  v39 = _mm_loadu_si128(a1 + 4);
  v40 = _mm_loadu_si128(a1 + 5);
  v41 = _mm_loadu_si128(a1 + 6);
  v42 = _mm_loadu_si128(a1 + 7);
  v43 = _mm_loadu_si128(a1 + 8);
  v44 = _mm_loadu_si128(a1 + 9);
  v45 = _mm_loadu_si128(a1 + 10);
  v46 = _mm_loadu_si128(a1 + 11);
  v3 = a1[12].m128i_i64[0];
  v47 = _mm_loadu_si128(a1 + 12);
  v48[0] = _mm_loadu_si128(a1 + 13);
  v48[1] = _mm_loadu_si128(a1 + 14);
  if ( ((v3 & 0x4000000000000000LL) == 0 || !HIBYTE(v48[0])) && !v38.m128i_i64[1] )
  {
    if ( byte_5057707 != 8 )
      goto LABEL_51;
    v36 = a1[9].m128i_i64[0];
    if ( (v47.m128i_i8[7] & 0x40) != 0 )
      HIBYTE(v48[0]) = 0;
    v38.m128i_i64[1] = (__int64)&v36;
  }
  if ( (a1[12].m128i_i8[7] & 0x40) != 0 )
    a1[13].m128i_i8[15] = 0;
  a1[3].m128i_i64[1] = 0;
  v4 = *(_DWORD *)(a2 + 320);
  if ( v4 != 1 )
  {
    if ( v4 != 2 )
      abort();
    v8 = 0;
    v9 = 0;
    v10 = *(char **)(a2 + 312);
    do
    {
      v11 = *v10++;
      v12 = (unsigned __int64)(v11 & 0x7F) << v9;
      v9 += 7;
      v8 |= v12;
    }
    while ( v11 < 0 );
    v7 = sub_39F6040(v10, (unsigned __int64)&v10[v8], (__int64)v37, 0);
    goto LABEL_17;
  }
  v5 = *(_QWORD *)(a2 + 304);
  if ( (int)v5 > 17 )
    goto LABEL_51;
  v6 = (_QWORD *)*((_QWORD *)v37 + (int)v5);
  if ( (v47.m128i_i8[7] & 0x40) != 0 && *((_BYTE *)v48 + (int)v5 + 8) )
    goto LABEL_12;
  if ( byte_5057700[(int)v5] != 8 )
LABEL_51:
    abort();
  v6 = (_QWORD *)*v6;
LABEL_12:
  v7 = (__int64)v6 + *(_QWORD *)(a2 + 296);
LABEL_17:
  v13 = v7;
  v14 = a2;
  v15 = byte_5057700;
  v16 = &a1[13].m128i_i8[8];
  v32 = a1;
  v17 = (__int64 *)a1;
  a1[9].m128i_i64[0] = v7;
  v18 = &a1[14].m128i_i8[10];
  while ( 2 )
  {
    switch ( *(_DWORD *)(v14 + 8) )
    {
      case 1:
        v24 = v13 + *(_QWORD *)v14;
        if ( (v32[12].m128i_i8[7] & 0x40) != 0 )
          goto LABEL_36;
        goto LABEL_26;
      case 2:
        v31 = (int)*(_QWORD *)v14;
        if ( *((_BYTE *)v48 + v31 + 8) )
        {
          if ( (int)*(_QWORD *)v14 > 17 )
            goto LABEL_51;
          if ( (v47.m128i_i8[7] & 0x40) != 0 )
          {
            v24 = *((_QWORD *)v37 + v31);
          }
          else
          {
            if ( byte_5057700[v31] != 8 )
              goto LABEL_51;
            v24 = **((_QWORD **)v37 + v31);
          }
LABEL_22:
          if ( (unsigned __int8)*v15 > 8u )
            goto LABEL_51;
LABEL_25:
          *v16 = 1;
        }
        else
        {
          v24 = *((_QWORD *)v37 + v31);
          if ( (v32[12].m128i_i8[7] & 0x40) != 0 )
LABEL_36:
            *v16 = 0;
        }
LABEL_26:
        *v17 = v24;
LABEL_27:
        ++v16;
        v14 += 16;
        ++v17;
        ++v15;
        if ( v18 != v16 )
          continue;
        result = v32[12].m128i_i64[0] & 0x7FFFFFFFFFFFFFFFLL;
        if ( *(_BYTE *)(a2 + 371) )
          result = v32[12].m128i_i64[0] | 0x8000000000000000LL;
        v32[12].m128i_i64[0] = result;
        return result;
      case 3:
        v26 = *(char **)v14;
        v27 = 0;
        v28 = 0;
        do
        {
          v29 = *v26++;
          v30 = (unsigned __int64)(v29 & 0x7F) << v28;
          v28 += 7;
          v27 |= v30;
        }
        while ( v29 < 0 );
        v34 = v15;
        v24 = sub_39F6040(v26, (unsigned __int64)&v26[v27], (__int64)v37, v13);
        v15 = v34;
        if ( (v32[12].m128i_i8[7] & 0x40) != 0 )
          goto LABEL_36;
        goto LABEL_26;
      case 4:
        v24 = v13 + *(_QWORD *)v14;
        if ( (unsigned __int8)*v15 <= 8u )
          goto LABEL_25;
        goto LABEL_51;
      case 5:
        v19 = *(char **)v14;
        v20 = 0;
        v21 = 0;
        do
        {
          v22 = *v19++;
          v23 = (unsigned __int64)(v22 & 0x7F) << v21;
          v21 += 7;
          v20 |= v23;
        }
        while ( v22 < 0 );
        v33 = v15;
        v24 = sub_39F6040(v19, (unsigned __int64)&v19[v20], (__int64)v37, v13);
        v15 = v33;
        goto LABEL_22;
      default:
        goto LABEL_27;
    }
  }
}
