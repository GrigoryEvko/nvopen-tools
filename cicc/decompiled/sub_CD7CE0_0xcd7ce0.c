// Function: sub_CD7CE0
// Address: 0xcd7ce0
//
__int64 __fastcall sub_CD7CE0(__int64 a1, __int64 a2, __int64 a3, const __m128i **a4, unsigned __int8 a5)
{
  char v9; // al
  __int64 v10; // rcx
  const __m128i *v11; // rsi
  const __m128i *v12; // rax
  const __m128i *v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx
  int v16; // ebx
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // r14
  const __m128i *v21; // rsi
  unsigned __int64 v22; // rax
  const __m128i *v23; // rbx
  const __m128i *v24; // r13
  __m128i *v25; // rdi
  size_t v26; // r14
  size_t v27; // rsi
  __m128i *v28; // rax
  size_t v29; // rdx
  __int8 *v30; // r14
  const __m128i *v31; // r13
  __int64 v32; // r12
  __m128i *i; // rax
  __int64 v34; // [rsp+8h] [rbp-68h]
  int *v36; // [rsp+18h] [rbp-58h]
  char v37; // [rsp+26h] [rbp-4Ah] BYREF
  char v38; // [rsp+27h] [rbp-49h] BYREF
  __int64 v39; // [rsp+28h] [rbp-48h] BYREF
  __int64 v40; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v41[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
  {
    v11 = *(const __m128i **)(a3 + 8);
    v12 = *(const __m128i **)a3;
    v13 = *a4;
    if ( (const __m128i *)((char *)v11 - *(_QWORD *)a3) == (const __m128i *)((char *)a4[1] - (char *)*a4) )
    {
      if ( v12 == v11 )
      {
LABEL_50:
        v10 = 1;
      }
      else
      {
        while ( v12->m128i_i32[0] == v13->m128i_i32[0]
             && v12->m128i_i32[1] == v13->m128i_i32[1]
             && v12->m128i_i32[2] == v13->m128i_i32[2]
             && v12->m128i_i32[3] == v13->m128i_i32[3]
             && v12[1].m128i_i32[0] == v13[1].m128i_i32[0]
             && v12[1].m128i_i32[1] == v13[1].m128i_i32[1]
             && v12[1].m128i_i32[2] == v13[1].m128i_i32[2]
             && v12[1].m128i_i32[3] == v13[1].m128i_i32[3]
             && v12[2].m128i_i32[0] == v13[2].m128i_i32[0] )
        {
          v12 = (const __m128i *)((char *)v12 + 36);
          v13 = (const __m128i *)((char *)v13 + 36);
          if ( v11 == v12 )
            goto LABEL_50;
        }
        v10 = 0;
      }
    }
  }
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             a2,
             a5,
             v10,
             &v37,
             &v39);
  if ( !(_BYTE)result )
  {
    if ( !v37 || (const __m128i **)a3 == a4 )
      return result;
    v23 = a4[1];
    v24 = *a4;
    v25 = *(__m128i **)a3;
    v26 = (char *)v23 - (char *)*a4;
    v27 = *(_QWORD *)(a3 + 16) - *(_QWORD *)a3;
    if ( v26 > v27 )
    {
      if ( v26 )
      {
        if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v25, v27, v15);
        v32 = sub_22077B0(v26);
        v25 = *(__m128i **)a3;
        v27 = *(_QWORD *)(a3 + 16) - *(_QWORD *)a3;
      }
      else
      {
        v32 = 0;
      }
      for ( i = (__m128i *)v32; v23 != v24; i = (__m128i *)((char *)i + 36) )
      {
        if ( i )
        {
          *i = _mm_loadu_si128(v24);
          i[1] = _mm_loadu_si128(v24 + 1);
          i[2].m128i_i32[0] = v24[2].m128i_i32[0];
        }
        v24 = (const __m128i *)((char *)v24 + 36);
      }
      if ( v25 )
        j_j___libc_free_0(v25, v27);
      v30 = (__int8 *)(v32 + v26);
      *(_QWORD *)a3 = v32;
      *(_QWORD *)(a3 + 16) = v30;
      goto LABEL_56;
    }
    v28 = *(__m128i **)(a3 + 8);
    v29 = (char *)v28 - (char *)v25;
    if ( v26 > (char *)v28 - (char *)v25 )
    {
      if ( v29 )
      {
        memmove(v25, v24, v29);
        v23 = a4[1];
        v24 = *a4;
        v28 = *(__m128i **)(a3 + 8);
        v25 = *(__m128i **)a3;
        v29 = (size_t)v28 - *(_QWORD *)a3;
      }
      v31 = (const __m128i *)((char *)v24 + v29);
      if ( v23 != v31 )
      {
        do
        {
          if ( v28 )
          {
            *v28 = _mm_loadu_si128(v31);
            v28[1] = _mm_loadu_si128(v31 + 1);
            v28[2].m128i_i32[0] = v31[2].m128i_i32[0];
          }
          v31 = (const __m128i *)((char *)v31 + 36);
          v28 = (__m128i *)((char *)v28 + 36);
        }
        while ( v23 != v31 );
        v30 = (__int8 *)(*(_QWORD *)a3 + v26);
        goto LABEL_56;
      }
    }
    else if ( v23 != v24 )
    {
      memmove(v25, v24, v26);
      v25 = *(__m128i **)a3;
    }
    v30 = &v25->m128i_i8[v26];
LABEL_56:
    *(_QWORD *)(a3 + 8) = v30;
    return a3;
  }
  v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v16 = 954437177 * ((__int64)(*(_QWORD *)(a3 + 8) - *(_QWORD *)a3) >> 2);
  if ( v16 )
  {
    v17 = (unsigned int)(v16 - 1);
    v18 = 1;
    v19 = 0;
    v34 = v17 + 2;
    do
    {
      while ( 1 )
      {
        v20 = v19 * 4 + 36;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v18 - 1),
               &v40) )
        {
          break;
        }
        v19 += 9;
        if ( v34 == ++v18 )
          goto LABEL_33;
      }
      v21 = *(const __m128i **)a3;
      v22 = 0x8E38E38E38E38E39LL * ((__int64)(*(_QWORD *)(a3 + 8) - *(_QWORD *)a3) >> 2);
      if ( v22 <= v18 - 1 )
      {
        if ( v22 < v18 )
        {
          sub_CD7B00((const __m128i **)a3, v18 - v22);
          v21 = *(const __m128i **)a3;
        }
        else if ( v22 > v18 && *(const __m128i **)(a3 + 8) != (const __m128i *)&v21->m128i_i8[v20] )
        {
          *(_QWORD *)(a3 + 8) = (char *)v21 + v20;
        }
      }
      v36 = &v21->m128i_i32[v19];
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Offset32",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Offset64",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 1);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Offset128",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 2);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexOffset32",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 3);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexOffset64",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 4);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexOffset128",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 5);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexMaxOffset32",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 6);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexMaxOffset64",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 7);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "TexMaxOffset128",
             1,
             0,
             &v38,
             v41) )
      {
        sub_CCD060(a1, v36 + 8);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v41[0]);
      }
      v19 += 9;
      ++v18;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v40);
    }
    while ( v34 != v18 );
  }
LABEL_33:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
  return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v39);
}
