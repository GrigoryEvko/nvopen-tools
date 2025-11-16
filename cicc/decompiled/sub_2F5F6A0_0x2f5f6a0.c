// Function: sub_2F5F6A0
// Address: 0x2f5f6a0
//
_QWORD *__fastcall sub_2F5F6A0(__int64 a1, const char *a2, _DWORD *a3, int **a4, _QWORD *a5, __int64 a6)
{
  size_t v11; // rax
  int *v12; // rax
  int v13; // edx
  __int64 v14; // rax
  _QWORD *result; // rax
  _QWORD *v16; // r15
  int v17; // edx
  __int64 v18; // rsi
  const __m128i *v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // r11
  int v27; // esi
  __m128i *v28; // rdx
  __m128i v29; // xmm1
  __int8 v30; // cl
  __int64 v31; // rdi
  char *v32; // rbx
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  _QWORD v37[5]; // [rsp+20h] [rbp-60h] BYREF
  int v38; // [rsp+48h] [rbp-38h]
  char v39; // [rsp+4Ch] [rbp-34h]

  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = *a4;
  v13 = **a4;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v13;
  *(_DWORD *)(a1 + 152) = *v12;
  v14 = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = v14;
  result = *(_QWORD **)a6;
  v36 = *(_QWORD *)a6 + 40LL * *(unsigned int *)(a6 + 8);
  if ( *(_QWORD *)a6 != v36 )
  {
    v16 = *(_QWORD **)a6;
    v35 = a1 + 176;
    do
    {
      v17 = *((_DWORD *)v16 + 4);
      v18 = v16[3];
      v37[4] = &unk_4A2B2D8;
      v19 = (const __m128i *)v37;
      v20 = v16[4];
      v21 = *v16;
      v39 = 1;
      v22 = v16[1];
      v23 = *(unsigned int *)(a1 + 188);
      v38 = v17;
      v24 = *(unsigned int *)(a1 + 184);
      v37[2] = v18;
      v37[3] = v20;
      v25 = *(_QWORD *)(a1 + 176);
      v26 = v24 + 1;
      v37[0] = v21;
      v27 = v24;
      v37[1] = v22;
      if ( v24 + 1 > v23 )
      {
        if ( v25 > (unsigned __int64)v37 )
        {
          v33 = v22;
          v34 = v21;
LABEL_12:
          sub_2F5F5E0(v35, v26, v24, v25, v22, v21);
          v24 = *(unsigned int *)(a1 + 184);
          v25 = *(_QWORD *)(a1 + 176);
          v21 = v34;
          v22 = v33;
          v27 = *(_DWORD *)(a1 + 184);
          goto LABEL_4;
        }
        v33 = v22;
        v34 = v21;
        v24 = v25 + 48 * v24;
        if ( (unsigned __int64)v37 >= v24 )
          goto LABEL_12;
        v32 = (char *)v37 - v25;
        sub_2F5F5E0(v35, v26, v24, v25, v22, v21);
        v25 = *(_QWORD *)(a1 + 176);
        v24 = *(unsigned int *)(a1 + 184);
        v22 = v33;
        v21 = v34;
        v19 = (const __m128i *)&v32[v25];
        v27 = *(_DWORD *)(a1 + 184);
      }
LABEL_4:
      v28 = (__m128i *)(v25 + 48 * v24);
      if ( v28 )
      {
        v29 = _mm_loadu_si128(v19 + 1);
        *v28 = _mm_loadu_si128(v19);
        v28[1] = v29;
        v28[2].m128i_i32[2] = v19[2].m128i_i32[2];
        v30 = v19[2].m128i_i8[12];
        v28[2].m128i_i64[0] = (__int64)&unk_4A2B2D8;
        v28[2].m128i_i8[12] = v30;
        v27 = *(_DWORD *)(a1 + 184);
      }
      v31 = *(_QWORD *)(a1 + 168);
      v16 += 5;
      *(_DWORD *)(a1 + 184) = v27 + 1;
      result = sub_C52F90(v31, v21, v22);
    }
    while ( (_QWORD *)v36 != v16 );
  }
  return result;
}
