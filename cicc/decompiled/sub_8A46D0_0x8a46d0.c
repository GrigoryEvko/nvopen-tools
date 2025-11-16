// Function: sub_8A46D0
// Address: 0x8a46d0
//
__m128i *__fastcall sub_8A46D0(
        const __m128i *a1,
        __m128i *a2,
        __int64 a3,
        __int64 *a4,
        unsigned int a5,
        int *a6,
        __int64 a7,
        int a8,
        const __m128i *a9)
{
  const __m128i *v10; // r14
  __m128i **v12; // r10
  __int64 v13; // r9
  __int64 v14; // r8
  __int8 v15; // cl
  __int64 v16; // rdi
  _QWORD *v17; // rax
  __m128i *v18; // rdx
  __int8 v19; // al
  __int8 v20; // al
  __int8 v21; // al
  __m128i *v22; // r12
  __int64 v23; // rax
  const __m128i *v24; // rax
  const __m128i *v25; // r10
  const __m128i *v26; // rdi
  char v27; // bl
  __int8 v28; // al
  const __m128i *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  int v34; // eax
  int v35; // ebx
  const __m128i *v36; // rax
  bool v37; // zf
  int v38; // eax
  int v39; // eax
  __m128i *v40; // rax
  __m128i *v42; // rax
  char v43; // cl
  __m128i *v44; // rax
  __m128i *v47; // [rsp+18h] [rbp-88h]
  int v50; // [rsp+2Ch] [rbp-74h]
  __m128i **v51; // [rsp+38h] [rbp-68h]
  unsigned int v52; // [rsp+40h] [rbp-60h]
  unsigned int v53; // [rsp+44h] [rbp-5Ch]
  __m128i *v54; // [rsp+48h] [rbp-58h]
  const __m128i *v55; // [rsp+48h] [rbp-58h]
  __m128i *v56; // [rsp+48h] [rbp-58h]
  const __m128i *v57; // [rsp+48h] [rbp-58h]
  __m128i **v58; // [rsp+48h] [rbp-58h]
  __m128i **v59; // [rsp+48h] [rbp-58h]
  char v60; // [rsp+48h] [rbp-58h]
  __m128i **v61; // [rsp+48h] [rbp-58h]
  int v62; // [rsp+5Ch] [rbp-44h] BYREF
  unsigned __int64 v63; // [rsp+60h] [rbp-40h] BYREF
  const __m128i *v64; // [rsp+68h] [rbp-38h] BYREF

  if ( a1 )
  {
    v10 = a1;
    v50 = a5 & 0x82000;
    v12 = 0;
    v53 = a5 & 0x2000;
    v47 = 0;
    while ( 1 )
    {
      v62 = 0;
      if ( v53 )
        break;
      v58 = v12;
      *(_DWORD *)(a7 + 92) = 0;
      v38 = sub_869530(v10[5].m128i_i64[0], a3, a2, (__int64 *)&v63, a5, a7, &v62);
      v12 = v58;
      if ( v62 )
        *a6 = 1;
      if ( v38 )
        break;
      v22 = 0;
LABEL_21:
      if ( (v10[2].m128i_i8[1] & 1) == 0 )
        goto LABEL_31;
      if ( (a5 & 0x40) == 0 )
        goto LABEL_23;
      v59 = v12;
      v40 = (__m128i *)sub_72B0C0(v10->m128i_i64[1], &dword_4F077C8);
      v12 = v59;
      *v40 = _mm_loadu_si128(v10);
      v40[1] = _mm_loadu_si128(v10 + 1);
      v40[2] = _mm_loadu_si128(v10 + 2);
      v40[3] = _mm_loadu_si128(v10 + 3);
      v40[4] = _mm_loadu_si128(v10 + 4);
      v40[5].m128i_i64[0] = v10[5].m128i_i64[0];
      if ( v59 )
        *v59 = v40;
      else
        v47 = v40;
      if ( !v22 )
        v22 = v40;
      if ( (v10[2].m128i_i8[1] & 1) != 0 )
      {
LABEL_23:
        v23 = qword_4F601A0;
        if ( qword_4F601A0 )
        {
          qword_4F601A0 = *(_QWORD *)qword_4F601A0;
        }
        else
        {
          v61 = v12;
          v23 = sub_823970(32);
          v12 = v61;
        }
        *(_QWORD *)v23 = 0;
        *(_QWORD *)(v23 + 8) = 0;
        *(_QWORD *)(v23 + 16) = 0;
        *(_DWORD *)(v23 + 24) = 0;
        if ( v22 && (v22[2].m128i_i8[1] & 2) != 0 )
        {
          *(_QWORD *)(v23 + 8) = v22;
        }
        else if ( *(_DWORD *)(a7 + 92) )
        {
          *(_QWORD *)(v23 + 8) = v10;
        }
        *(_QWORD *)(v23 + 16) = v10;
        *(_DWORD *)(v23 + 24) = *(_DWORD *)(a7 + 72);
        if ( *(_QWORD *)a7 )
        {
          *(_QWORD *)v23 = **(_QWORD **)(a7 + 8);
          **(_QWORD **)(a7 + 8) = v23;
        }
        else
        {
          *(_QWORD *)a7 = v23;
        }
        *(_QWORD *)(a7 + 8) = v23;
LABEL_31:
        v10 = (const __m128i *)v10->m128i_i64[0];
        if ( !v10 )
          return v47;
      }
      else
      {
        v10 = (const __m128i *)v10->m128i_i64[0];
        if ( !v10 )
          return v47;
      }
    }
    v51 = v12;
    v22 = 0;
    while ( 1 )
    {
      v24 = (const __m128i *)sub_8D72A0(v10);
      v25 = v24;
      if ( a8 )
      {
        v64 = v24;
        --a8;
        v26 = v24;
        v27 = 0;
        v52 = 0;
      }
      else
      {
        if ( a9 )
        {
          v64 = a9;
          v52 = 0;
        }
        else
        {
          v57 = v24;
          v35 = *(_DWORD *)(a7 + 88);
          *(_DWORD *)(a7 + 88) = 0;
          v36 = (const __m128i *)sub_8A2270((__int64)v24, a2, a3, a4, a5, a6, (__m128i *)a7);
          v37 = (v10[2].m128i_i8[1] & 1) == 0;
          v52 = 0;
          v64 = v36;
          a9 = v36;
          v25 = v57;
          if ( !v37 )
          {
            v52 = *(_DWORD *)(a7 + 88);
            *(_DWORD *)(a7 + 88) = v35;
          }
        }
        if ( a9 == v25 )
        {
          v26 = a9;
          v27 = 0;
          a9 = 0;
        }
        else
        {
          v27 = 0;
          sub_645520((__int64 *)&v64);
          v29 = v64;
          if ( dword_4F0690C && v64[8].m128i_i8[12] == 12 )
          {
            v39 = sub_8D4C10(v64, 1);
            if ( v39 )
            {
              v60 = v39;
              v42 = sub_73D4C0(v64, dword_4F077C4 == 2);
              v43 = v60;
              v64 = v42;
              v29 = v42;
              if ( (v60 & 8) != 0 )
              {
                v44 = sub_73C570(v42, 8);
                v43 = v60;
                v64 = v44;
                v29 = v44;
              }
              v27 = v43 & 0x7F;
              if ( (v43 & 4) != 0 && unk_4F06908 )
              {
                v64 = sub_73C570(v29, 4);
                v29 = v64;
              }
            }
            else
            {
              v29 = v64;
            }
          }
          if ( (unsigned int)sub_8D9760(v29) )
          {
            v25 = a9;
            v26 = v64;
            a9 = 0;
            *a6 = 1;
          }
          else
          {
            v25 = a9;
            v26 = v64;
            a8 = 0;
            a9 = 0;
          }
        }
      }
      v55 = v25;
      v18 = (__m128i *)sub_72B0C0((__int64)v26, &dword_4F077C8);
      v18[1].m128i_i64[0] = (__int64)v55;
      v18[2].m128i_i32[0] = v18[2].m128i_i32[0] & 0xFFFC07FF | ((v27 & 0x7F) << 11);
      v18[1].m128i_i64[1] = v10[1].m128i_i64[1];
      v18[2].m128i_i32[1] = v10[2].m128i_i32[1];
      v28 = v10[2].m128i_i8[1];
      if ( (v28 & 1) == 0 )
      {
        v18[2].m128i_i8[1] = v18[2].m128i_i8[1] & 0xFD | v28 & 2;
        goto LABEL_9;
      }
      v13 = v53;
      if ( v53 )
      {
        v15 = v28 & 2 | v18[2].m128i_i8[1] & 0xFD;
        v18[2].m128i_i8[1] = v15;
        if ( !v52 )
          goto LABEL_9;
      }
      else
      {
        v14 = v52;
        if ( !v52 )
        {
          v18[2].m128i_i8[1] |= 2u;
          goto LABEL_9;
        }
        v15 = v18[2].m128i_i8[1];
      }
      v16 = v10[5].m128i_i64[0];
      v18[2].m128i_i8[1] = v15 | 1;
      if ( *(_QWORD *)(a7 + 32) || *(_QWORD *)(a7 + 56) )
      {
        v54 = v18;
        v17 = sub_892C00(v16, (__int64 *)a7);
        v18 = v54;
        v16 = (__int64)v17;
      }
      v18[5].m128i_i64[0] = v16;
LABEL_9:
      v19 = v10[2].m128i_i8[0];
      if ( (v19 & 4) != 0 )
      {
        v20 = v18[2].m128i_i8[0] | 4;
        v18[2].m128i_i8[0] = v20;
        v18[2].m128i_i8[0] = v10[2].m128i_i8[0] & 8 | v20 & 0xF7;
        v19 = v10[2].m128i_i8[0];
      }
      if ( (v19 & 0x10) != 0 )
      {
        v18[2].m128i_i8[0] |= 0x10u;
        v18[3].m128i_i64[0] = v10[3].m128i_i64[0];
      }
      if ( v50 )
      {
        v21 = v10[2].m128i_i8[0] & 0x80 | v18[2].m128i_i8[0] & 0x7F;
        v18[2].m128i_i8[0] = v21;
        v18[2].m128i_i8[0] = v10[2].m128i_i8[0] & 0x40 | v21 & 0xBF;
      }
      if ( v51 )
        *v51 = v18;
      else
        v47 = v18;
      if ( !v22 )
        v22 = v18;
      if ( v53
        || (v56 = v18,
            sub_867630(v63, 0, (__int64)v18, 0, v14, v13),
            v34 = sub_866C00(v63, 0, v30, v31, v32, v33),
            v18 = v56,
            !v34) )
      {
        v12 = (__m128i **)v18;
        goto LABEL_21;
      }
      v51 = (__m128i **)v56;
    }
  }
  return 0;
}
