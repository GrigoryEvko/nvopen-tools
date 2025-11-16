// Function: sub_3545A70
// Address: 0x3545a70
//
__int64 __fastcall sub_3545A70(__int64 a1, const __m128i *a2, _QWORD *a3)
{
  unsigned int v3; // ecx
  int v4; // r14d
  _QWORD *v5; // r12
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  unsigned __int64 *v8; // rdi
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r13
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  const __m128i *v22; // r9
  unsigned __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i *v27; // rax
  __m128i v28; // xmm1
  unsigned __int64 *v29; // rax
  unsigned __int64 v31; // [rsp+8h] [rbp-1C8h]
  _QWORD *v33; // [rsp+20h] [rbp-1B0h]
  __int64 v35; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned __int64 *v36; // [rsp+38h] [rbp-198h]
  __int64 v37; // [rsp+40h] [rbp-190h]
  int v38; // [rsp+48h] [rbp-188h]
  char v39; // [rsp+4Ch] [rbp-184h]
  char v40; // [rsp+50h] [rbp-180h] BYREF
  unsigned __int64 *v41; // [rsp+90h] [rbp-140h] BYREF
  __int64 v42; // [rsp+98h] [rbp-138h]
  _OWORD v43[19]; // [rsp+A0h] [rbp-130h] BYREF

  v3 = 1;
  v4 = 0x7FFFFFFF;
  v5 = (_QWORD *)(a1 + 40);
  v6 = _mm_loadu_si128(a2);
  v7 = _mm_loadu_si128(a2 + 1);
  v36 = (unsigned __int64 *)&v40;
  v35 = 0;
  v37 = 8;
  v38 = 0;
  v39 = 1;
  v41 = (unsigned __int64 *)v43;
  v42 = 0x800000001LL;
  v8 = (unsigned __int64 *)v43;
  v43[0] = v6;
  v43[1] = v7;
  while ( v3 )
  {
    while ( 1 )
    {
      v9 = v3--;
      v10 = v8[4 * v9 - 3];
      LODWORD(v42) = v3;
      v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v39 )
      {
        if ( sub_C8CA60((__int64)&v35, v11) )
          goto LABEL_29;
        goto LABEL_7;
      }
      v12 = v36;
      v13 = &v36[HIDWORD(v37)];
      if ( v36 != v13 )
        break;
LABEL_7:
      v14 = *(_QWORD **)(a1 + 48);
      if ( v14 )
      {
        v15 = v5;
        do
        {
          while ( 1 )
          {
            v16 = v14[2];
            v17 = v14[3];
            if ( v14[4] >= v11 )
              break;
            v14 = (_QWORD *)v14[3];
            if ( !v17 )
              goto LABEL_12;
          }
          v15 = v14;
          v14 = (_QWORD *)v14[2];
        }
        while ( v16 );
LABEL_12:
        if ( v15 != v5 && v15[4] <= v11 )
        {
          if ( v4 > *((_DWORD *)v15 + 10) )
            v4 = *((_DWORD *)v15 + 10);
          v18 = sub_35459D0(a3, v11);
          v21 = *(_QWORD *)v18;
          v22 = (const __m128i *)(*(_QWORD *)v18 + 32LL * *(unsigned int *)(v18 + 8));
          if ( v22 != *(const __m128i **)v18 )
          {
            v33 = v5;
            v23 = *(_QWORD *)v18;
            v24 = *(_QWORD *)v18 + 32LL * *(unsigned int *)(v18 + 8);
            do
            {
              while ( 1 )
              {
                v25 = (*(__int64 *)(v23 + 8) >> 1) & 3;
                if ( v25 == 2 || v25 == 3 )
                  break;
                v23 += 32LL;
                if ( v23 == v24 )
                  goto LABEL_23;
              }
              v26 = (unsigned int)v42;
              v20 = HIDWORD(v42);
              v22 = (const __m128i *)v23;
              v19 = v41;
              if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
              {
                if ( (unsigned __int64)v41 > v23 || v23 >= (unsigned __int64)&v41[4 * (unsigned int)v42] )
                {
                  sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 0x20u, v21, v23);
                  v19 = v41;
                  v26 = (unsigned int)v42;
                  v22 = (const __m128i *)v23;
                }
                else
                {
                  v31 = v23 - (_QWORD)v41;
                  sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 0x20u, v21, v23);
                  v19 = v41;
                  v22 = (const __m128i *)((char *)v41 + v31);
                  v26 = (unsigned int)v42;
                }
              }
              v23 += 32LL;
              v27 = (__m128i *)&v19[4 * v26];
              *v27 = _mm_loadu_si128(v22);
              v28 = _mm_loadu_si128(v22 + 1);
              LODWORD(v42) = v42 + 1;
              v27[1] = v28;
            }
            while ( v23 != v24 );
LABEL_23:
            v5 = v33;
          }
          if ( !v39 )
            goto LABEL_38;
          v29 = v36;
          v20 = HIDWORD(v37);
          v19 = &v36[HIDWORD(v37)];
          if ( v36 == v19 )
          {
LABEL_42:
            if ( HIDWORD(v37) < (unsigned int)v37 )
            {
              ++HIDWORD(v37);
              *v19 = v11;
              v3 = v42;
              ++v35;
              v8 = v41;
              goto LABEL_30;
            }
LABEL_38:
            sub_C8CC70((__int64)&v35, v11, (__int64)v19, v20, v21, (__int64)v22);
            v3 = v42;
            v8 = v41;
            goto LABEL_30;
          }
          while ( v11 != *v29 )
          {
            if ( v19 == ++v29 )
              goto LABEL_42;
          }
        }
      }
LABEL_29:
      v3 = v42;
      v8 = v41;
LABEL_30:
      if ( !v3 )
        goto LABEL_31;
    }
    while ( v11 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_7;
    }
  }
LABEL_31:
  if ( v8 != (unsigned __int64 *)v43 )
    _libc_free((unsigned __int64)v8);
  if ( !v39 )
    _libc_free((unsigned __int64)v36);
  return (unsigned int)v4;
}
