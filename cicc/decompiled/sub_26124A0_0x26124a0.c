// Function: sub_26124A0
// Address: 0x26124a0
//
_QWORD *__fastcall sub_26124A0(
        __int64 a1,
        char a2,
        __int64 a3,
        int a4,
        int a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        int a12)
{
  unsigned __int64 *v12; // r12
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  int v19; // eax
  _QWORD *result; // rax
  __int64 v21; // rdi
  char *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  char *v25; // rsi
  __int64 v26; // r13
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  char *v29; // rsi
  __int64 v30; // r13
  _QWORD *v31; // rdi
  char *v32; // rsi
  _QWORD v33[5]; // [rsp+8h] [rbp-28h] BYREF

  v12 = (unsigned __int64 *)(a1 + 104);
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v15 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)(a1 + 84) = a3;
  v16 = _mm_loadu_si128((const __m128i *)&a9);
  v17 = _mm_loadu_si128((const __m128i *)&a10);
  *(_DWORD *)(a1 + 92) = a4;
  v18 = _mm_loadu_si128((const __m128i *)&a11);
  v19 = a12;
  *(_DWORD *)(a1 + 96) = a5;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 80) = v19;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(__m128i *)a1 = v14;
  *(__m128i *)(a1 + 16) = v15;
  *(__m128i *)(a1 + 32) = v16;
  *(__m128i *)(a1 + 48) = v17;
  *(__m128i *)(a1 + 64) = v18;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  if ( !a2 )
    goto LABEL_2;
  v23 = sub_22077B0(0x28u);
  v24 = v23;
  if ( v23 )
  {
    *(_QWORD *)(v23 + 8) = 0;
    *(_BYTE *)(v23 + 16) = 1;
    *(_DWORD *)(v23 + 20) = 0;
    *(_QWORD *)v23 = &unk_4A0ECB8;
    *(_QWORD *)(v23 + 24) = 0;
    *(_QWORD *)(v23 + 32) = 0;
  }
  v33[0] = v23;
  v25 = *(char **)(a1 + 112);
  if ( v25 == *(char **)(a1 + 120) )
  {
    sub_235A6C0(v12, v25, v33);
    v24 = v33[0];
LABEL_29:
    if ( v24 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
    goto LABEL_14;
  }
  if ( !v25 )
  {
    *(_QWORD *)(a1 + 112) = 8;
    goto LABEL_29;
  }
  *(_QWORD *)v25 = v23;
  *(_QWORD *)(a1 + 112) += 8LL;
LABEL_14:
  if ( !(_BYTE)qword_4FF2848 )
    goto LABEL_2;
  v26 = sub_C5F790(v24, (__int64)v25);
  v27 = (_QWORD *)sub_22077B0(0x10u);
  v28 = v27;
  if ( v27 )
  {
    v27[1] = v26;
    *v27 = &unk_4A1F6F0;
  }
  v33[0] = v27;
  v29 = *(char **)(a1 + 112);
  if ( v29 == *(char **)(a1 + 120) )
  {
    sub_235A6C0(v12, v29, v33);
    v28 = (_QWORD *)v33[0];
  }
  else
  {
    if ( v29 )
    {
      *(_QWORD *)v29 = v27;
      *(_QWORD *)(a1 + 112) += 8LL;
      goto LABEL_2;
    }
    *(_QWORD *)(a1 + 112) = 8;
  }
  if ( v28 )
    (*(void (__fastcall **)(_QWORD *))(*v28 + 8LL))(v28);
LABEL_2:
  result = (_QWORD *)sub_22077B0(0x28u);
  v21 = (__int64)result;
  if ( result )
  {
    result[1] = 0;
    *((_BYTE *)result + 16) = 0;
    result = &unk_4A0ECB8;
    *(_DWORD *)(v21 + 20) = 0;
    *(_QWORD *)v21 = &unk_4A0ECB8;
    *(_QWORD *)(v21 + 24) = 0;
    *(_QWORD *)(v21 + 32) = 0;
  }
  v33[0] = v21;
  v22 = *(char **)(a1 + 112);
  if ( v22 == *(char **)(a1 + 120) )
  {
    result = (_QWORD *)sub_235A6C0(v12, v22, v33);
    v21 = v33[0];
  }
  else
  {
    if ( v22 )
    {
      *(_QWORD *)v22 = v21;
      *(_QWORD *)(a1 + 112) += 8LL;
      goto LABEL_7;
    }
    *(_QWORD *)(a1 + 112) = 8;
  }
  if ( v21 )
    result = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
LABEL_7:
  if ( (_BYTE)qword_4FF2848 )
  {
    v30 = sub_C5F790(v21, (__int64)v22);
    result = (_QWORD *)sub_22077B0(0x10u);
    v31 = result;
    if ( result )
    {
      result[1] = v30;
      result = &unk_4A1F6F0;
      *v31 = &unk_4A1F6F0;
    }
    v33[0] = v31;
    v32 = *(char **)(a1 + 112);
    if ( v32 == *(char **)(a1 + 120) )
    {
      result = (_QWORD *)sub_235A6C0(v12, v32, v33);
      v31 = (_QWORD *)v33[0];
    }
    else
    {
      if ( v32 )
      {
        *(_QWORD *)v32 = v31;
        *(_QWORD *)(a1 + 112) += 8LL;
        return result;
      }
      *(_QWORD *)(a1 + 112) = 8;
    }
    if ( v31 )
      return (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *))(*v31 + 8LL))(v31);
  }
  return result;
}
