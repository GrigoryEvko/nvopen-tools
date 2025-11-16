// Function: sub_2864A10
// Address: 0x2864a10
//
void __fastcall sub_2864A10(
        const __m128i *a1,
        const __m128i *a2,
        unsigned int a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  unsigned int *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rcx
  __m128i *v13; // rax
  const __m128i *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // bl
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  __m128i v25; // xmm1
  __int64 v26; // rdx
  __int64 v27; // rdx
  __m128i v28; // xmm2
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned int v31; // r9d
  char v32; // dl
  char v33; // r8
  __int64 *v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rsi
  unsigned int v40; // edx
  unsigned __int64 v41; // rsi
  _QWORD *v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // [rsp+8h] [rbp-118h]
  __int64 v47; // [rsp+10h] [rbp-110h]
  __int64 v48; // [rsp+10h] [rbp-110h]
  __int64 v50; // [rsp+20h] [rbp-100h] BYREF
  char v51; // [rsp+28h] [rbp-F8h] BYREF
  unsigned int v52[3]; // [rsp+2Ch] [rbp-F4h] BYREF
  __int64 v53; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v54; // [rsp+40h] [rbp-E0h]
  __int64 v55; // [rsp+48h] [rbp-D8h]
  const __m128i *v56[6]; // [rsp+50h] [rbp-D0h] BYREF
  _BYTE v57[24]; // [rsp+80h] [rbp-A0h] BYREF
  char v58; // [rsp+98h] [rbp-88h]
  __int64 v59; // [rsp+A0h] [rbp-80h]
  unsigned __int64 v60[2]; // [rsp+A8h] [rbp-78h] BYREF
  _BYTE v61[32]; // [rsp+B8h] [rbp-68h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-48h]
  __m128i v63; // [rsp+E0h] [rbp-40h]

  v52[0] = a3;
  v56[3] = (const __m128i *)&v51;
  v56[4] = (const __m128i *)&v50;
  v10 = v52;
  v50 = a6;
  v51 = a7;
  v56[0] = (const __m128i *)a4;
  v56[1] = a1;
  v56[2] = a2;
  v56[5] = (const __m128i *)v52;
  if ( a7 )
  {
    v11 = a4[11];
  }
  else
  {
    v10 = (unsigned int *)v50;
    v11 = *(_QWORD *)(a4[5] + 8 * v50);
  }
  v53 = v11;
  v12 = a1[4].m128i_u32[2];
  if ( (_DWORD)v12
    || a2[2].m128i_i32[0] != 2
    || *(_WORD *)(v11 + 24) != 8
    || (v38 = sub_D33D80((_QWORD *)v11, a1->m128i_i64[1], (__int64)v10, v12, a5), *(_WORD *)(v38 + 24)) )
  {
LABEL_5:
    v13 = *(__m128i **)a5;
    v14 = (const __m128i *)(*(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8));
    goto LABEL_6;
  }
  v39 = *(_QWORD *)(v38 + 32);
  v40 = *(_DWORD *)(v39 + 32);
  v41 = *(_QWORD *)(v39 + 24);
  if ( v40 > 0x40 )
  {
    v42 = *(_QWORD **)v41;
LABEL_28:
    v46 = (__int64)v42;
    goto LABEL_29;
  }
  v42 = (_QWORD *)v41;
  if ( ((1LL << ((unsigned __int8)v40 - 1)) & v41) == 0 )
    goto LABEL_28;
  v46 = 0;
  if ( v40 )
    v46 = (__int64)(v41 << (64 - (unsigned __int8)v40)) >> (64 - (unsigned __int8)v40);
LABEL_29:
  v13 = *(__m128i **)a5;
  v14 = *(const __m128i **)a5;
  a6 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
  if ( a6 != *(_QWORD *)a5 )
  {
    do
    {
      *(__m128i *)v57 = _mm_loadu_si128(v14);
      if ( !v14->m128i_i8[8] )
      {
        v43 = v14->m128i_i64[0] - v46;
        v57[8] = 0;
        *(_QWORD *)v57 = v43;
        v47 = a6;
        sub_2864680(v56, v53, v43, *(__int64 *)&v57[8], a5, a6);
        a6 = v47;
      }
      ++v14;
    }
    while ( (const __m128i *)a6 != v14 );
    goto LABEL_5;
  }
LABEL_6:
  v15 = (__int64 *)v13;
  if ( v13 != v14 )
  {
    do
    {
      v16 = *v15;
      v17 = v15[1];
      v15 += 2;
      *(_QWORD *)v57 = v16;
      *(_QWORD *)&v57[8] = v17;
      sub_2864680(v56, v53, v16, v17, a5, a6);
    }
    while ( v14 != (const __m128i *)v15 );
  }
  v18 = sub_28579B0((__int64)&v53, (__int64 *)a1->m128i_i64[1]);
  v55 = v19;
  v20 = v19;
  v54 = v18;
  if ( !sub_D968A0(v53) )
  {
    v23 = v54;
    if ( v54 )
    {
      if ( !a4[1] || v20 == *((_BYTE *)a4 + 16) )
      {
        v24 = *a4;
        v25 = _mm_loadu_si128((const __m128i *)(a4 + 1));
        v60[0] = (unsigned __int64)v61;
        *(_QWORD *)v57 = v24;
        LOBYTE(v24) = *((_BYTE *)a4 + 24);
        v60[1] = 0x400000000LL;
        v58 = v24;
        v26 = a4[4];
        *(__m128i *)&v57[8] = v25;
        v59 = v26;
        v27 = *((unsigned int *)a4 + 12);
        if ( (_DWORD)v27 )
        {
          v48 = v54;
          sub_2850210((__int64)v60, (__int64)(a4 + 5), v27, 0x400000000LL, v21, v22);
          v23 = v48;
        }
        v28 = _mm_loadu_si128((const __m128i *)a4 + 6);
        v29 = a2[45].m128i_i64[1];
        v30 = a2[44].m128i_i64[1];
        v31 = a2[2].m128i_u32[0];
        v62 = a4[11];
        if ( v57[16] )
          v20 = 1;
        *(_QWORD *)&v57[8] += v23;
        v32 = a2[45].m128i_i8[0];
        v33 = a2[46].m128i_i8[0];
        v63 = v28;
        v34 = (__int64 *)a1[3].m128i_i64[0];
        v35 = a2[3].m128i_u32[0];
        v57[16] = v20;
        if ( sub_2850770(v34, v30, v32, v29, v33, v31, a2[2].m128i_i64[1], v35, (__int64)v57) )
        {
          if ( v51 )
          {
            v62 = v53;
          }
          else
          {
            v44 = v50;
            v45 = v60[0];
            *(_QWORD *)(v60[0] + 8 * v50) = v53;
            sub_2857080((__int64)v57, a1[3].m128i_i64[1], v45, v44, v36, v37);
          }
          sub_2862B30((__int64)a1, (__int64)a2, v52[0], (unsigned __int64)v57, v36, v37);
        }
        if ( (_BYTE *)v60[0] != v61 )
          _libc_free(v60[0]);
      }
    }
  }
}
