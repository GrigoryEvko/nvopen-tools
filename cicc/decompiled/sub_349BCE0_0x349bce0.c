// Function: sub_349BCE0
// Address: 0x349bce0
//
void __fastcall sub_349BCE0(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        unsigned __int64 a5,
        unsigned __int64 *a6,
        __m128i a7,
        _QWORD *a8)
{
  char v8; // dl
  __int64 v10; // r8
  int v11; // eax
  __int32 *v12; // rcx
  __int64 v13; // rsi
  int v14; // r10d
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // r10d
  _QWORD *v19; // rsi
  __int16 *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r15
  unsigned __int16 v23; // r9
  __int64 v24; // r14
  _QWORD *v25; // rax
  __m128i *v26; // rsi
  __int32 v27; // edx
  __m128i *v28; // rsi
  _QWORD *v29; // rax
  __m128i *v30; // rsi
  __int32 v31; // edx
  __int64 v32; // r10
  __int32 v33; // ecx
  __int64 v34; // r13
  unsigned int v35; // eax
  __int64 v36; // r8
  unsigned int v37; // r15d
  __int64 *v38; // rax
  __int64 v39; // r14
  __int64 v40; // rsi
  unsigned __int8 *v41; // rax
  __int32 v42; // edx
  unsigned __int16 v43; // [rsp+4h] [rbp-6Ch]
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  __m128i v46; // [rsp+10h] [rbp-60h] BYREF
  __int64 v47; // [rsp+20h] [rbp-50h] BYREF
  int v48; // [rsp+28h] [rbp-48h]
  __m128i v49; // [rsp+30h] [rbp-40h] BYREF

  v46.m128i_i64[0] = a2;
  v46.m128i_i64[1] = a3;
  if ( a5 > 1 )
    return;
  v8 = *a4;
  if ( (unsigned __int8)(*a4 - 88) > 0x1Bu || ((1LL << (v8 - 88)) & 0x8420001) == 0 )
    return;
  v10 = 0;
  while ( 1 )
  {
    v11 = *(_DWORD *)(v46.m128i_i64[0] + 24);
    if ( v11 == 11 || v11 == 35 )
    {
      if ( v8 == 115 )
        goto LABEL_20;
      v34 = *(_QWORD *)(v46.m128i_i64[0] + 96);
      v45 = v10;
      v35 = sub_3289F80(a1, 8, 0);
      v36 = v45;
      v37 = *(_DWORD *)(v34 + 32);
      if ( v37 == 1 )
      {
        if ( v35 > 2 )
          BUG();
        if ( v35 == 1 )
        {
          v39 = *(_QWORD *)(v34 + 24);
          goto LABEL_46;
        }
        v38 = *(__int64 **)(v34 + 24);
      }
      else
      {
        v38 = *(__int64 **)(v34 + 24);
        if ( v37 > 0x40 )
        {
          v39 = *v38;
LABEL_46:
          v40 = *(_QWORD *)(v46.m128i_i64[0] + 80);
          v47 = v40;
          if ( v40 )
          {
            sub_B96E90((__int64)&v47, v40, 1);
            v36 = v45;
          }
          v48 = *(_DWORD *)(v46.m128i_i64[0] + 72);
          v41 = sub_3400BD0((__int64)a8, v36 + v39, (__int64)&v47, 8, 0, 1u, a7, 0);
          v26 = (__m128i *)a6[1];
          v49.m128i_i64[0] = (__int64)v41;
          v49.m128i_i32[2] = v42;
          if ( v26 == (__m128i *)a6[2] )
            goto LABEL_52;
          if ( v26 )
          {
LABEL_26:
            *v26 = _mm_loadu_si128(&v49);
            v26 = (__m128i *)a6[1];
          }
LABEL_27:
          a6[1] = (unsigned __int64)&v26[1];
LABEL_28:
          if ( v47 )
            sub_B91220((__int64)&v47, v47);
          return;
        }
        v39 = 0;
        if ( !v37 )
          goto LABEL_46;
      }
      v39 = (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
      goto LABEL_46;
    }
    if ( v8 == 110 )
      goto LABEL_9;
LABEL_20:
    if ( (unsigned int)(v11 - 13) <= 1 || (unsigned int)(v11 - 37) <= 1 )
    {
      v20 = *(__int16 **)(v46.m128i_i64[0] + 48);
      v21 = *(_QWORD *)(v46.m128i_i64[0] + 80);
      v22 = *(_QWORD *)(v46.m128i_i64[0] + 104);
      v23 = *v20;
      v24 = *((_QWORD *)v20 + 1);
      v47 = v21;
      if ( v21 )
      {
        v43 = v23;
        v44 = v10;
        sub_B96E90((__int64)&v47, v21, 1);
        v23 = v43;
        v10 = v44;
      }
      v48 = *(_DWORD *)(v46.m128i_i64[0] + 72);
      v25 = sub_33ED290((__int64)a8, *(_QWORD *)(v46.m128i_i64[0] + 96), (__int64)&v47, v23, v24, v22 + v10, 1, 0);
      v26 = (__m128i *)a6[1];
      v49.m128i_i64[0] = (__int64)v25;
      v49.m128i_i32[2] = v27;
      if ( v26 != (__m128i *)a6[2] )
      {
        if ( v26 )
          goto LABEL_26;
        goto LABEL_27;
      }
LABEL_52:
      sub_337C620(a6, v26, &v49);
      goto LABEL_28;
    }
    if ( v11 == 43 || v11 == 19 )
      break;
    if ( v11 == 6 )
    {
      v28 = (__m128i *)a6[1];
      if ( v28 == (__m128i *)a6[2] )
      {
        sub_33764F0(a6, v28, &v46);
      }
      else
      {
        if ( v28 )
          *v28 = _mm_load_si128(&v46);
        a6[1] += 16LL;
      }
      return;
    }
LABEL_9:
    if ( (unsigned int)(v11 - 56) > 1 )
      return;
    v12 = *(__int32 **)(v46.m128i_i64[0] + 40);
    v13 = *(_QWORD *)v12;
    v14 = *(_DWORD *)(*(_QWORD *)v12 + 24LL);
    if ( v14 == 35 || v14 == 11 )
    {
      v32 = *((_QWORD *)v12 + 5);
      v33 = v12[12];
      v46.m128i_i64[0] = v32;
      v46.m128i_i32[2] = v33;
      v16 = 2LL * (v11 == 56) - 1;
    }
    else
    {
      if ( v11 != 56 )
        return;
      v13 = *((_QWORD *)v12 + 5);
      v15 = *(_DWORD *)(v13 + 24);
      if ( v15 != 11 && v15 != 35 )
        return;
      v46.m128i_i64[0] = *(_QWORD *)v12;
      v46.m128i_i32[2] = v12[2];
      v16 = 1;
    }
    v17 = *(_QWORD *)(v13 + 96);
    v18 = *(_DWORD *)(v17 + 32);
    v19 = *(_QWORD **)(v17 + 24);
    if ( v18 > 0x40 )
    {
      v10 += *v19 * v16;
    }
    else if ( v18 )
    {
      v10 += ((__int64)((_QWORD)v19 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18)) * v16;
    }
  }
  v29 = sub_33F0E20(
          a8,
          *(_QWORD *)(v46.m128i_i64[0] + 96),
          **(unsigned __int16 **)(v46.m128i_i64[0] + 48),
          *(_QWORD *)(*(_QWORD *)(v46.m128i_i64[0] + 48) + 8LL),
          *(_QWORD *)(v46.m128i_i64[0] + 104) + v10,
          1,
          *(_DWORD *)(v46.m128i_i64[0] + 112));
  v30 = (__m128i *)a6[1];
  v49.m128i_i64[0] = (__int64)v29;
  v49.m128i_i32[2] = v31;
  if ( v30 == (__m128i *)a6[2] )
  {
    sub_337C620(a6, v30, &v49);
  }
  else
  {
    if ( v30 )
    {
      *v30 = _mm_loadu_si128(&v49);
      v30 = (__m128i *)a6[1];
    }
    a6[1] = (unsigned __int64)&v30[1];
  }
}
