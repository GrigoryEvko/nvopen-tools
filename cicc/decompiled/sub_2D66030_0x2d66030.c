// Function: sub_2D66030
// Address: 0x2d66030
//
__int64 __fastcall sub_2D66030(__int64 a1, char *a2, __int64 a3, unsigned int a4)
{
  const __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r8
  __m128i v13; // xmm2
  __int64 v14; // rax
  unsigned int v15; // r14d
  __m128i *v16; // rax
  const __m128i *v17; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r13
  unsigned __int64 v22; // r8
  unsigned int v23; // ebx
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rdx
  const __m128i *v29; // rax
  __int64 v30; // rbx
  unsigned int v31; // edx
  __int64 *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rax
  __m128i *v37; // rax
  __int64 *v38; // [rsp+0h] [rbp-100h]
  unsigned __int64 v39; // [rsp+0h] [rbp-100h]
  __int64 v40; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v41; // [rsp+10h] [rbp-F0h] BYREF
  unsigned int v42; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v43; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE *v44; // [rsp+28h] [rbp-D8h]
  char v45; // [rsp+30h] [rbp-D0h]
  __int64 *v46; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v47; // [rsp+48h] [rbp-B8h] BYREF
  unsigned int v48; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v49; // [rsp+60h] [rbp-A0h]
  const void *v50; // [rsp+68h] [rbp-98h] BYREF
  unsigned int v51; // [rsp+70h] [rbp-90h]
  char v52; // [rsp+78h] [rbp-88h]
  _BYTE v53[72]; // [rsp+80h] [rbp-80h] BYREF

  if ( a3 != 1 )
  {
    if ( !a3 )
      return 1;
    v6 = *(const __m128i **)(a1 + 96);
    v7 = v6[1].m128i_i64[1];
    if ( v7 && a2 != (char *)v6[3].m128i_i64[0] )
      return 0;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = v7 + a3;
    v10 = *(_QWORD *)(a1 + 72);
    v11 = *(_QWORD *)(a1 + 24);
    *(__m128i *)v53 = _mm_loadu_si128(v6);
    v12 = *(unsigned int *)(a1 + 80);
    *(__m128i *)&v53[16] = _mm_loadu_si128(v6 + 1);
    v13 = _mm_loadu_si128(v6 + 2);
    *(_QWORD *)&v53[24] = v9;
    *(__m128i *)&v53[32] = v13;
    *(__m128i *)&v53[48] = _mm_loadu_si128(v6 + 3);
    v14 = v6[4].m128i_i64[0];
    *(_QWORD *)&v53[48] = a2;
    *(_QWORD *)&v53[64] = v14;
    v15 = (*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64, _QWORD))(*(_QWORD *)v8 + 1288LL))(
            v8,
            v11,
            v53,
            v10,
            v12,
            0);
    if ( !(_BYTE)v15 )
      return 0;
    v16 = *(__m128i **)(a1 + 96);
    *v16 = _mm_loadu_si128((const __m128i *)v53);
    v16[1] = _mm_loadu_si128((const __m128i *)&v53[16]);
    v16[2] = _mm_loadu_si128((const __m128i *)&v53[32]);
    v16[3] = _mm_loadu_si128((const __m128i *)&v53[48]);
    v16[4].m128i_i8[0] = v53[64];
    if ( *a2 == 42
      && (v40 = *((_QWORD *)a2 - 8)) != 0
      && (v30 = *((_QWORD *)a2 - 4), *(_BYTE *)v30 == 17)
      && !sub_2D59C50(a2, *(_QWORD *)(a1 + 32))
      && *(_DWORD *)(v30 + 32) + 1 - (unsigned int)sub_969260(v30 + 24) <= 0x40 )
    {
      v53[64] = 0;
      *(_QWORD *)&v53[48] = v40;
      v31 = *(_DWORD *)(v30 + 32);
      v32 = *(__int64 **)(v30 + 24);
      if ( v31 > 0x40 )
      {
        v33 = *v32;
      }
      else
      {
        v33 = 0;
        if ( v31 )
          v33 = (__int64)((_QWORD)v32 << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31);
      }
      v34 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)&v53[8] += *(_QWORD *)&v53[24] * v33;
      v35 = (*(__int64 (__fastcall **)(__int64, _QWORD, _BYTE *, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v34 + 1288LL))(
              v34,
              *(_QWORD *)(a1 + 24),
              v53,
              *(_QWORD *)(a1 + 72),
              *(unsigned int *)(a1 + 80),
              0);
      if ( (_BYTE)v35 )
      {
        v15 = v35;
        sub_9C95B0(*(_QWORD *)a1, (__int64)a2);
        qmemcpy(*(void **)(a1 + 96), v53, 0x41u);
        return v15;
      }
      v17 = *(const __m128i **)(a1 + 96);
      *(__m128i *)v53 = _mm_loadu_si128(v17);
      *(__m128i *)&v53[16] = _mm_loadu_si128(v17 + 1);
      *(__m128i *)&v53[32] = _mm_loadu_si128(v17 + 2);
      *(__m128i *)&v53[48] = _mm_loadu_si128(v17 + 3);
      v53[64] = v17[4].m128i_i8[0];
    }
    else
    {
      v17 = *(const __m128i **)(a1 + 96);
    }
    if ( !v17->m128i_i64[1] )
      return v15;
    if ( *a2 != 84 )
      return v15;
    sub_2D59A60((__int64)&v43, (__int64)a2, *(_QWORD *)(a1 + 32));
    if ( !v45 )
      return v15;
    v19 = *v43;
    if ( (unsigned __int8)v19 <= 0x36u )
    {
      v20 = 0x40540000000000LL;
      if ( _bittest64(&v20, v19) )
      {
        if ( ((v43[1] >> 1) & 2) != 0 || (v43[1] & 2) != 0 )
          return v15;
      }
    }
    if ( *v44 != 17 )
      return v15;
    v46 = (__int64 *)v43;
    sub_9865C0((__int64)&v47, (__int64)(v44 + 24));
    v21 = (__int64)v46;
    v52 = 1;
    v49 = (unsigned __int8 *)v46;
    v51 = v48;
    v50 = v47;
    v42 = v48;
    if ( v48 > 0x40 )
    {
      sub_C43780((__int64)&v41, &v50);
      v22 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 24LL);
      LODWORD(v47) = v42;
      if ( v42 > 0x40 )
      {
        v39 = v22;
        sub_C43780((__int64)&v46, (const void **)&v41);
        v22 = v39;
        goto LABEL_23;
      }
    }
    else
    {
      v41 = v47;
      v22 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 24LL);
      LODWORD(v47) = v48;
    }
    v46 = (__int64 *)v41;
LABEL_23:
    sub_C47170((__int64)&v46, v22);
    v23 = (unsigned int)v47;
    LODWORD(v44) = (_DWORD)v47;
    v43 = (unsigned __int8 *)v46;
    v38 = v46;
    if ( v23 + 1 - (unsigned int)sub_969260((__int64)&v43) <= 0x40 )
    {
      v53[64] = 0;
      v24 = (__int64)v38;
      *(_QWORD *)&v53[48] = v21;
      if ( v23 > 0x40 )
      {
        v25 = sub_C444A0((__int64)&v43);
        v24 = -1;
        if ( v23 - v25 <= 0x40 )
          v24 = *v38;
      }
      v26 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)&v53[8] -= v24;
      v27 = *(_QWORD *)(a1 + 24);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _BYTE *, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v26 + 1288LL))(
             v26,
             v27,
             v53,
             *(_QWORD *)(a1 + 72),
             *(unsigned int *)(a1 + 80),
             0) )
      {
        if ( !*(_QWORD *)(a1 + 56) )
          sub_4263D6(v26, v27, v28);
        v36 = (*(__int64 (__fastcall **)(__int64))(a1 + 64))(a1 + 40);
        if ( (unsigned __int8)sub_B19DB0(v36, v21, *(_QWORD *)(a1 + 88)) )
        {
          sub_9C95B0(*(_QWORD *)a1, v21);
          v37 = *(__m128i **)(a1 + 96);
          *v37 = _mm_loadu_si128((const __m128i *)v53);
          v37[1] = _mm_loadu_si128((const __m128i *)&v53[16]);
          v37[2] = _mm_loadu_si128((const __m128i *)&v53[32]);
          v37[3] = _mm_loadu_si128((const __m128i *)&v53[48]);
          v37[4].m128i_i8[0] = v53[64];
          sub_969240((__int64 *)&v43);
          sub_969240((__int64 *)&v41);
          if ( !v52 )
            return v15;
          goto LABEL_30;
        }
      }
      v29 = *(const __m128i **)(a1 + 96);
      *(__m128i *)v53 = _mm_loadu_si128(v29);
      *(__m128i *)&v53[16] = _mm_loadu_si128(v29 + 1);
      *(__m128i *)&v53[32] = _mm_loadu_si128(v29 + 2);
      *(__m128i *)&v53[48] = _mm_loadu_si128(v29 + 3);
      v53[64] = v29[4].m128i_i8[0];
    }
    sub_969240((__int64 *)&v43);
    sub_969240((__int64 *)&v41);
    if ( v52 )
    {
LABEL_30:
      v52 = 0;
      sub_969240((__int64 *)&v50);
    }
    return v15;
  }
  return sub_2D65BF0(a1, (unsigned __int8 *)a2, a4);
}
