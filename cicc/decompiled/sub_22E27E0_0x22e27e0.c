// Function: sub_22E27E0
// Address: 0x22e27e0
//
__int64 *__fastcall sub_22E27E0(__int64 a1, const char *a2, _DWORD **a3, _DWORD *a4, _QWORD *a5, __int64 a6)
{
  size_t v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 *result; // rax
  __int64 *v14; // r15
  __int64 v15; // rsi
  const __m128i *v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // r8
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r11
  int v24; // esi
  __m128i *v25; // rdx
  __m128i v26; // xmm1
  __int8 v27; // cl
  __int64 v28; // rdi
  char *v29; // rbx
  _DWORD *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  _QWORD v36[4]; // [rsp+20h] [rbp-60h] BYREF
  void *v37; // [rsp+40h] [rbp-40h]
  int v38; // [rsp+48h] [rbp-38h]
  char v39; // [rsp+4Ch] [rbp-34h]

  v10 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v10);
  if ( *(_QWORD *)(a1 + 136) )
  {
    v11 = sub_CEADF0();
    v36[0] = "cl::location(x) specified more than once!";
    LOWORD(v37) = 259;
    sub_C53280(a1, (__int64)v36, 0, 0, (__int64)v11);
  }
  else
  {
    v30 = *a3;
    *(_BYTE *)(a1 + 156) = 1;
    *(_QWORD *)(a1 + 136) = v30;
    *(_DWORD *)(a1 + 152) = *v30;
  }
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = v12;
  result = *(__int64 **)a6;
  v35 = *(_QWORD *)a6 + 40LL * *(unsigned int *)(a6 + 8);
  if ( *(_QWORD *)a6 != v35 )
  {
    v14 = *(__int64 **)a6;
    v33 = a1 + 176;
    do
    {
      v15 = v14[3];
      v16 = (const __m128i *)v36;
      v17 = v14[4];
      v18 = *v14;
      v19 = v14[1];
      v20 = *(unsigned int *)(a1 + 188);
      v38 = *((_DWORD *)v14 + 4);
      v21 = *(unsigned int *)(a1 + 184);
      v36[2] = v15;
      v36[3] = v17;
      v22 = *(_QWORD *)(a1 + 176);
      v23 = v21 + 1;
      v36[0] = v18;
      v24 = v21;
      v36[1] = v19;
      v37 = &unk_4A09FD0;
      v39 = 1;
      if ( v21 + 1 > v20 )
      {
        if ( v22 > (unsigned __int64)v36 )
        {
          v31 = v19;
          v32 = v18;
LABEL_14:
          sub_22E2720(v33, v23, v21, v22, v19, v18);
          v21 = *(unsigned int *)(a1 + 184);
          v22 = *(_QWORD *)(a1 + 176);
          v18 = v32;
          v19 = v31;
          v24 = *(_DWORD *)(a1 + 184);
          goto LABEL_6;
        }
        v31 = v19;
        v32 = v18;
        v21 = v22 + 48 * v21;
        if ( (unsigned __int64)v36 >= v21 )
          goto LABEL_14;
        v29 = (char *)v36 - v22;
        sub_22E2720(v33, v23, v21, v22, v19, v18);
        v22 = *(_QWORD *)(a1 + 176);
        v21 = *(unsigned int *)(a1 + 184);
        v19 = v31;
        v18 = v32;
        v16 = (const __m128i *)&v29[v22];
        v24 = *(_DWORD *)(a1 + 184);
      }
LABEL_6:
      v25 = (__m128i *)(v22 + 48 * v21);
      if ( v25 )
      {
        v26 = _mm_loadu_si128(v16 + 1);
        *v25 = _mm_loadu_si128(v16);
        v25[1] = v26;
        v25[2].m128i_i32[2] = v16[2].m128i_i32[2];
        v27 = v16[2].m128i_i8[12];
        v25[2].m128i_i64[0] = (__int64)&unk_4A09FD0;
        v25[2].m128i_i8[12] = v27;
        v24 = *(_DWORD *)(a1 + 184);
      }
      v28 = *(_QWORD *)(a1 + 168);
      v14 += 5;
      *(_DWORD *)(a1 + 184) = v24 + 1;
      result = sub_C52F90(v28, v18, v19);
    }
    while ( (__int64 *)v35 != v14 );
  }
  return result;
}
