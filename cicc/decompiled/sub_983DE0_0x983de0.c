// Function: sub_983DE0
// Address: 0x983de0
//
__int64 __fastcall sub_983DE0(__int64 a1, const char *a2, _DWORD *a3, _QWORD *a4, int **a5, __int64 a6)
{
  size_t v10; // rax
  __int64 v11; // rax
  int *v12; // rax
  int v13; // edx
  __int64 result; // rax
  __int64 v15; // r14
  __int64 v16; // r15
  int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // r8
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  const __m128i *v24; // rbx
  unsigned __int64 v25; // rcx
  __int64 v26; // r11
  int v27; // esi
  __m128i *v28; // rdx
  __m128i v29; // xmm1
  __int64 v30; // rdi
  char *v31; // rbx
  __int64 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  _QWORD v36[5]; // [rsp+20h] [rbp-60h] BYREF
  int v37; // [rsp+48h] [rbp-38h]
  char v38; // [rsp+4Ch] [rbp-34h]

  v10 = strlen(a2);
  sub_C53080(a1, a2, v10);
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v11 = a4[1];
  *(_QWORD *)(a1 + 40) = *a4;
  *(_QWORD *)(a1 + 48) = v11;
  v12 = *a5;
  v13 = **a5;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v13;
  *(_DWORD *)(a1 + 152) = *v12;
  result = *(_QWORD *)a6;
  v15 = *(_QWORD *)a6 + 40LL * *(unsigned int *)(a6 + 8);
  if ( *(_QWORD *)a6 != v15 )
  {
    v16 = *(_QWORD *)a6;
    v34 = a1 + 176;
    do
    {
      v17 = *(_DWORD *)(v16 + 16);
      v18 = *(_QWORD *)(v16 + 24);
      v36[4] = &unk_49D9580;
      v19 = *(_QWORD *)(v16 + 32);
      v20 = *(_QWORD *)v16;
      v38 = 1;
      v21 = *(_QWORD *)(v16 + 8);
      v22 = *(unsigned int *)(a1 + 188);
      v37 = v17;
      v23 = *(unsigned int *)(a1 + 184);
      v24 = (const __m128i *)v36;
      v36[2] = v18;
      v36[3] = v19;
      v25 = *(_QWORD *)(a1 + 176);
      v26 = v23 + 1;
      v36[0] = v20;
      v27 = v23;
      v36[1] = v21;
      if ( v23 + 1 > v22 )
      {
        if ( v25 > (unsigned __int64)v36 )
        {
          v32 = v21;
          v33 = v20;
LABEL_12:
          sub_983D20(v34, v26);
          v23 = *(unsigned int *)(a1 + 184);
          v25 = *(_QWORD *)(a1 + 176);
          v20 = v33;
          v21 = v32;
          v27 = *(_DWORD *)(a1 + 184);
          goto LABEL_4;
        }
        v32 = v21;
        v33 = v20;
        if ( (unsigned __int64)v36 >= v25 + 48 * v23 )
          goto LABEL_12;
        v31 = (char *)v36 - v25;
        sub_983D20(v34, v26);
        v25 = *(_QWORD *)(a1 + 176);
        v23 = *(unsigned int *)(a1 + 184);
        v21 = v32;
        v20 = v33;
        v24 = (const __m128i *)&v31[v25];
        v27 = *(_DWORD *)(a1 + 184);
      }
LABEL_4:
      v28 = (__m128i *)(v25 + 48 * v23);
      if ( v28 )
      {
        v29 = _mm_loadu_si128(v24 + 1);
        *v28 = _mm_loadu_si128(v24);
        v28[1] = v29;
        v28[2].m128i_i32[2] = v24[2].m128i_i32[2];
        v25 = v24[2].m128i_u8[12];
        v28[2].m128i_i64[0] = (__int64)&unk_49D9580;
        v28[2].m128i_i8[12] = v25;
        v27 = *(_DWORD *)(a1 + 184);
      }
      v30 = *(_QWORD *)(a1 + 168);
      v16 += 40;
      *(_DWORD *)(a1 + 184) = v27 + 1;
      result = sub_C52F90(v30, v20, v21, v25);
    }
    while ( v15 != v16 );
  }
  return result;
}
