// Function: sub_16D8B50
// Address: 0x16d8b50
//
double *__fastcall sub_16D8B50(
        __m128i **a1,
        unsigned __int8 *a2,
        size_t a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 *src,
        size_t n,
        _BYTE *a9,
        double *a10)
{
  double *result; // rax
  __int64 v13; // r14
  unsigned int v14; // r8d
  _QWORD *v15; // r10
  __int64 v16; // r9
  __int64 v17; // rax
  __m128i *v18; // r13
  __int64 v19; // rax
  unsigned int v20; // r8d
  _QWORD *v21; // r10
  _QWORD *v22; // rcx
  _BYTE *v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // r14
  _QWORD *v30; // [rsp+0h] [rbp-80h]
  _QWORD *v31; // [rsp+0h] [rbp-80h]
  unsigned int v32; // [rsp+8h] [rbp-78h]
  _QWORD *v33; // [rsp+8h] [rbp-78h]
  _QWORD *v34; // [rsp+8h] [rbp-78h]
  _QWORD *v35; // [rsp+10h] [rbp-70h]
  unsigned int v36; // [rsp+10h] [rbp-70h]
  unsigned int v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+20h] [rbp-60h]
  __int64 v40; // [rsp+30h] [rbp-50h]

  result = a10;
  if ( !(_BYTE)a6 )
  {
    *a1 = 0;
    return result;
  }
  if ( !qword_4FA1400 )
    sub_16C1EA0((__int64)&qword_4FA1400, (__int64 (*)(void))sub_16D5D90, (__int64)sub_16D95E0, a4, a5, a6);
  v13 = qword_4FA1400;
  if ( !qword_4FA1610 )
    sub_16C1EA0((__int64)&qword_4FA1610, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  v39 = qword_4FA1610;
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30C0((pthread_mutex_t **)v39);
  else
    ++*(_DWORD *)(v39 + 8);
  v14 = sub_16D19C0(v13, src, n);
  v15 = (_QWORD *)(*(_QWORD *)v13 + 8LL * v14);
  v16 = *v15;
  if ( !*v15 )
  {
LABEL_19:
    v30 = v15;
    v32 = v14;
    v19 = malloc(n + 49);
    v20 = v32;
    v21 = v30;
    v22 = (_QWORD *)v19;
    if ( !v19 )
    {
      if ( n == -49 )
      {
        v26 = malloc(1u);
        v20 = v32;
        v21 = v30;
        v22 = 0;
        if ( v26 )
        {
          v23 = (_BYTE *)(v26 + 48);
          v22 = (_QWORD *)v26;
          goto LABEL_30;
        }
      }
      v31 = v22;
      v34 = v21;
      v36 = v20;
      sub_16BD1C0("Allocation failed", 1u);
      v20 = v36;
      v21 = v34;
      v22 = v31;
    }
    v23 = v22 + 6;
    if ( n + 1 <= 1 )
    {
LABEL_21:
      v23[n] = 0;
      *v22 = n;
      v22[1] = 0;
      v22[2] = 0;
      v22[3] = 0;
      v22[4] = 0xA800000000LL;
      *v21 = v22;
      ++*(_DWORD *)(v13 + 12);
      v24 = (__int64 *)(*(_QWORD *)v13 + 8LL * (unsigned int)sub_16D1CD0(v13, v20));
      v16 = *v24;
      if ( !*v24 || v16 == -8 )
      {
        v25 = v24 + 1;
        do
        {
          do
            v16 = *v25++;
          while ( !v16 );
        }
        while ( v16 == -8 );
      }
      goto LABEL_11;
    }
LABEL_30:
    v33 = v22;
    v35 = v21;
    v37 = v20;
    v27 = memcpy(v23, src, n);
    v22 = v33;
    v21 = v35;
    v20 = v37;
    v23 = v27;
    goto LABEL_21;
  }
  if ( v16 == -8 )
  {
    --*(_DWORD *)(v13 + 16);
    goto LABEL_19;
  }
LABEL_11:
  if ( !*(_QWORD *)(v16 + 8) )
  {
    v38 = v16;
    v28 = (__int64 *)sub_22077B0(112);
    v16 = v38;
    v29 = v28;
    if ( v28 )
    {
      sub_16D7E20(v28, src, n, a9, (__int64)a10);
      v16 = v38;
    }
    *(_QWORD *)(v16 + 8) = v29;
  }
  v40 = v16;
  v17 = *(_QWORD *)sub_16D8970(v16 + 16, a2, a3);
  v18 = (__m128i *)(v17 + 8);
  if ( !*(_QWORD *)(v17 + 144) )
    sub_16D8060(v17 + 8, (__int64)a2, a3, a4, a5, *(_QWORD *)(v40 + 8));
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30E0((pthread_mutex_t **)v39);
  else
    --*(_DWORD *)(v39 + 8);
  *a1 = v18;
  return sub_16D7910(v18);
}
