// Function: sub_2ABBD40
// Address: 0x2abbd40
//
__m128i *__fastcall sub_2ABBD40(__m128i *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  unsigned __int64 v5; // rbx
  __int64 v7; // r13
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // r9d
  int v15; // edx
  char v16; // al
  const __m128i *v17; // r11
  __int64 *v18; // rsi
  int v19; // ecx
  const __m128i *v20; // r15
  _BYTE *v21; // rax
  unsigned int v22; // eax
  __m128i v23; // xmm7
  __m128i v24; // xmm6
  __int64 v25; // rax
  _QWORD *v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v30; // [rsp+18h] [rbp-C8h]
  __int64 v31; // [rsp+28h] [rbp-B8h]
  unsigned int v32; // [rsp+34h] [rbp-ACh]
  unsigned int v33; // [rsp+3Ch] [rbp-A4h]
  __int64 v34; // [rsp+48h] [rbp-98h]
  _BYTE *v35; // [rsp+58h] [rbp-88h]
  unsigned int v36; // [rsp+60h] [rbp-80h]
  __int32 v37; // [rsp+60h] [rbp-80h]
  const __m128i *v38; // [rsp+60h] [rbp-80h]
  __int64 *v40; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v41; // [rsp+78h] [rbp-68h]
  __m128i v42; // [rsp+80h] [rbp-60h] BYREF
  __m128i v43; // [rsp+90h] [rbp-50h] BYREF
  __m128i v44[4]; // [rsp+A0h] [rbp-40h] BYREF

  v5 = HIDWORD(a3);
  v42 = (__m128i)1uLL;
  v43 = 0u;
  v44[0] = 0u;
  if ( !byte_500ED88 )
    goto LABEL_3;
  LODWORD(v7) = *(_DWORD *)(*(_QWORD *)(a2 + 48) + 96LL);
  if ( (_DWORD)v7 )
    goto LABEL_3;
  v36 = a3;
  if ( !sub_2AB4060((__int64 *)a2) )
    goto LABEL_3;
  if ( (unsigned int)qword_500ECA8 <= 1 )
  {
    v12 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a2 + 32LL) + 72LL);
    if ( (unsigned __int8)sub_B2D610(v12, 47)
      || (unsigned __int8)sub_B2D610(v12, 18)
      || (unsigned __int8)sub_B2D610(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)a2 + 32LL) + 72LL), 18)
      || !sub_2AB42B0(*(_QWORD *)(a2 + 48), a3, a4) )
    {
      goto LABEL_3;
    }
    v13 = *(_QWORD *)(a2 + 48);
    v14 = v36;
    v15 = *(_DWORD *)(v13 + 4);
    v16 = *(_BYTE *)(v13 + 8);
    v32 = v36;
    if ( (_BYTE)v5 && v16 )
      v32 = v36 * v15;
    v17 = *(const __m128i **)(a2 + 136);
    v18 = *(__int64 **)(*(_QWORD *)(a2 + 64) + 112LL);
    v34 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 336LL);
    v38 = &v17[3 * *(unsigned int *)(a2 + 144)];
    if ( v17 == v38 )
    {
LABEL_33:
      v23 = _mm_loadu_si128(&v43);
      *a1 = _mm_loadu_si128(&v42);
      v24 = _mm_loadu_si128(v44);
      a1[1] = v23;
      a1[2] = v24;
      return a1;
    }
    v19 = a4;
    v20 = *(const __m128i **)(a2 + 136);
    v35 = 0;
    v31 = v14 * v19;
    v33 = v14;
    while ( 1 )
    {
      v40 = (__int64 *)v20->m128i_i64[0];
      if ( sub_2ABB960(a2 + 88, (__int64 *)&v40) )
      {
        v22 = v20->m128i_i32[0];
        if ( v20->m128i_i8[4] )
        {
          if ( v33 > v22 )
            goto LABEL_22;
        }
        else
        {
          if ( !(_BYTE)v5 )
          {
            if ( v33 < v22 )
              goto LABEL_26;
            if ( !v35 )
            {
              v25 = sub_2BF1320(a2, v20->m128i_i64[0]);
              v7 = sub_2C46ED0(*(_QWORD *)(v25 + 200), v18);
              v26 = sub_DA2C50((__int64)v18, v34, v31, 0);
              v35 = sub_DCFA50(v18, v7, (__int64)v26);
              v27 = sub_DA2C50((__int64)v18, v34, (unsigned int)(v31 - 1), 0);
              v29 = v29 & 0xFFFFFF0000000000LL | 0x24;
              LODWORD(v7) = v31 - 1;
              if ( (unsigned __int8)sub_DC3A60((__int64)v18, v29, v35, v27) )
              {
                v28 = sub_DBB9F0((__int64)v18, (__int64)v35, 0, 0);
                sub_AB0910((__int64)&v40, v28);
                if ( v41 <= 0x40 )
                  LODWORD(v7) = (_DWORD)v40;
                else
                  v7 = *v40;
                sub_969240((__int64 *)&v40);
              }
            }
            v21 = sub_DA2C50((__int64)v18, v34, v20->m128i_u32[0], 0);
            v30 = v30 & 0xFFFFFF0000000000LL | 0x22;
            if ( (unsigned __int8)sub_DC3A60((__int64)v18, v30, v21, v35) )
              goto LABEL_26;
LABEL_22:
            if ( !v42.m128i_i8[4] && v42.m128i_i32[0] == 1 || sub_2AB3D90(a2, (__int64)v20, (__int64)&v42, v7) )
            {
              v42 = _mm_loadu_si128(v20);
              v43 = _mm_loadu_si128(v20 + 1);
              v44[0] = _mm_loadu_si128(v20 + 2);
            }
            goto LABEL_26;
          }
          if ( v32 > v22 )
            goto LABEL_22;
        }
      }
LABEL_26:
      v20 += 3;
      if ( v38 == v20 )
        goto LABEL_33;
    }
  }
  LODWORD(v40) = qword_500ECA8;
  v37 = qword_500ECA8;
  BYTE4(v40) = 0;
  if ( sub_2ABB960(a2 + 88, (__int64 *)&v40) )
  {
    a1->m128i_i8[4] = 0;
    a1->m128i_i64[1] = 0;
    a1->m128i_i32[0] = v37;
    a1[1].m128i_i32[0] = 0;
    a1[1].m128i_i64[1] = 0;
    a1[2].m128i_i32[0] = 0;
    a1[2].m128i_i32[2] = 0;
    a1[2].m128i_i8[12] = 0;
    return a1;
  }
LABEL_3:
  v8 = _mm_loadu_si128(&v43);
  v9 = _mm_loadu_si128(v44);
  *a1 = _mm_loadu_si128(&v42);
  a1[1] = v8;
  a1[2] = v9;
  return a1;
}
