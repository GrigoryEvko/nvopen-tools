// Function: sub_1DA8B50
// Address: 0x1da8b50
//
__int64 __fastcall sub_1DA8B50(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        char a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // rax
  _QWORD *i; // rdx
  __int64 v14; // r15
  __int64 v15; // r15
  const __m128i *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rcx
  int v20; // r12d
  __int64 result; // rax
  unsigned __int32 v22; // r12d
  __int16 v23; // ax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // r8d
  int v28; // r9d
  unsigned int v29; // eax
  __int64 v30; // rdx
  unsigned __int64 v31; // rdx
  unsigned int v32; // r9d
  __int64 *v33; // rax
  __int64 v34; // r11
  int v35; // eax
  __int16 v36; // ax
  __int64 v37; // rax
  int v38; // edi
  __int64 v39; // [rsp-10h] [rbp-C0h]
  int v40; // [rsp+Ch] [rbp-A4h]
  _QWORD *v41; // [rsp+10h] [rbp-A0h]
  int v43; // [rsp+20h] [rbp-90h]
  unsigned int v44; // [rsp+30h] [rbp-80h]
  __int64 v45; // [rsp+30h] [rbp-80h]
  unsigned __int64 v47; // [rsp+38h] [rbp-78h]
  __int64 v48; // [rsp+48h] [rbp-68h] BYREF
  __m128i v49; // [rsp+50h] [rbp-60h] BYREF
  __m128i v50; // [rsp+60h] [rbp-50h]
  __int64 v51; // [rsp+70h] [rbp-40h]

  v44 = a5 & 0x7FFFFFFF;
  v43 = a2;
  v10 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a7 + 272) + 392LL) + 16LL * *(unsigned int *)(a2 + 48));
  v11 = v10[1];
  v12 = (_QWORD *)*v10;
  if ( (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11 >> 1) & 3) >= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(a4 >> 1) & 3) )
    v11 = a4;
  for ( i = (_QWORD *)(a3 & 0xFFFFFFFFFFFFFFF8LL); ; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( i )
    {
      v14 = i[2];
      if ( v14 )
        break;
    }
    if ( v12 == i )
    {
      v15 = sub_1DD5E10(a2, *(_QWORD *)(a2 + 32));
      goto LABEL_9;
    }
  }
  v36 = *(_WORD *)(v14 + 46);
  if ( (v36 & 4) == 0 && (v36 & 8) != 0 )
    LOBYTE(v37) = sub_1E15D00(v14, 64, 1);
  else
    v37 = (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 8LL) >> 6) & 1LL;
  if ( (_BYTE)v37 )
  {
    v15 = sub_1DD5EE0(a2);
    if ( (a5 & 0x7FFFFFFF) == 0x7FFFFFFF )
      goto LABEL_59;
LABEL_10:
    v16 = (const __m128i *)(a1[5] + 40LL * v44);
    v49 = _mm_loadu_si128(v16);
    v50 = _mm_loadu_si128(v16 + 1);
    v51 = v16[2].m128i_i64[0];
    goto LABEL_11;
  }
  if ( (*(_BYTE *)v14 & 4) != 0 )
  {
    v15 = *(_QWORD *)(v14 + 8);
  }
  else
  {
    while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
      v14 = *(_QWORD *)(v14 + 8);
    v15 = *(_QWORD *)(v14 + 8);
  }
LABEL_9:
  if ( (a5 & 0x7FFFFFFF) != 0x7FFFFFFF )
    goto LABEL_10;
LABEL_59:
  v49 = (__m128i)0x800000000uLL;
  v50 = 0u;
  v51 = 0;
LABEL_11:
  v41 = (_QWORD *)a1[1];
  if ( a6 )
  {
    v40 = 1;
    if ( (a5 & 0x80000000) != 0 )
      v41 = (_QWORD *)sub_15C48E0(v41, 1, 0, 0, 0);
  }
  else
  {
    v40 = a5 >> 31;
  }
  v47 = v11 & 0xFFFFFFFFFFFFFFF8LL;
  v17 = a2 + 24;
LABEL_14:
  v18 = a1[2];
  v19 = *a1;
  v20 = *(_QWORD *)(a8 + 8) + 768;
  v48 = v18;
  if ( v18 )
  {
    v45 = v19;
    sub_1623A60((__int64)&v48, v18, 2);
    v19 = v45;
  }
  sub_1E1C3C0(v43, v15, (unsigned int)&v48, v20, v40, (unsigned int)&v49, v19, (__int64)v41);
  result = v39;
  if ( v48 )
    result = sub_161E7C0((__int64)&v48, v48);
  if ( !v49.m128i_i8[0] )
  {
    v22 = v49.m128i_u32[2];
    if ( v17 != v15 )
    {
      v23 = *(_WORD *)(v15 + 46);
      if ( (v23 & 4) != 0 )
        goto LABEL_35;
LABEL_22:
      if ( (v23 & 8) != 0 )
      {
        result = sub_1E15D00(v15, 64, 1);
        goto LABEL_24;
      }
LABEL_35:
      while ( 1 )
      {
        result = (*(_QWORD *)(*(_QWORD *)(v15 + 16) + 8LL) >> 6) & 1LL;
LABEL_24:
        if ( (_BYTE)result )
          break;
        v24 = *(_QWORD *)(a7 + 272);
        v25 = *(unsigned int *)(v24 + 384);
        if ( (_DWORD)v25 )
        {
          v26 = *(_QWORD *)(v24 + 368);
          v27 = v25 - 1;
          v28 = 1;
          v29 = (v25 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v30 = *(_QWORD *)(v26 + 16LL * v29);
          if ( v15 == v30 )
          {
LABEL_27:
            v31 = v15;
            if ( (*(_BYTE *)(v15 + 46) & 4) != 0 )
            {
              do
                v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v31 + 46) & 4) != 0 );
            }
            v32 = v27 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v33 = (__int64 *)(v26 + 16LL * v32);
            v34 = *v33;
            if ( v31 != *v33 )
            {
              v35 = 1;
              while ( v34 != -8 )
              {
                v38 = v35 + 1;
                v32 = v27 & (v35 + v32);
                v33 = (__int64 *)(v26 + 16LL * v32);
                v34 = *v33;
                if ( *v33 == v31 )
                  goto LABEL_30;
                v35 = v38;
              }
              v33 = (__int64 *)(v26 + 16 * v25);
            }
LABEL_30:
            result = v33[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_DWORD *)(result + 24) >= *(_DWORD *)(v47 + 24) )
              return result;
          }
          else
          {
            while ( v30 != -8 )
            {
              v29 = v27 & (v28 + v29);
              v30 = *(_QWORD *)(v26 + 16LL * v29);
              if ( v15 == v30 )
                goto LABEL_27;
              ++v28;
            }
          }
        }
        result = sub_1E16810(v15, v22, 0, 0, a9);
        if ( (_DWORD)result != -1 )
        {
          if ( (*(_BYTE *)v15 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v15 + 46) & 8) != 0 )
              v15 = *(_QWORD *)(v15 + 8);
          }
          v15 = *(_QWORD *)(v15 + 8);
          if ( v17 != v15 )
            goto LABEL_14;
          return result;
        }
        if ( (*(_BYTE *)v15 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v15 + 46) & 8) != 0 )
            v15 = *(_QWORD *)(v15 + 8);
        }
        v15 = *(_QWORD *)(v15 + 8);
        if ( v17 == v15 )
          return result;
        v23 = *(_WORD *)(v15 + 46);
        if ( (v23 & 4) == 0 )
          goto LABEL_22;
      }
    }
  }
  return result;
}
