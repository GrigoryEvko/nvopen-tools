// Function: sub_23ADC70
// Address: 0x23adc70
//
unsigned __int64 __fastcall sub_23ADC70(__int64 a1, const char *a2, int **a3, _DWORD *a4, __int64 *a5, __int64 *a6)
{
  size_t v9; // rax
  int **v10; // r11
  int *v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
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
  char *v33; // rbx
  __int64 v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  _QWORD v40[5]; // [rsp+20h] [rbp-60h] BYREF
  int v41; // [rsp+48h] [rbp-38h]
  char v42; // [rsp+4Ch] [rbp-34h]

  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 156) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A15AF0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)a1 = &unk_4A15B60;
  *(_QWORD *)(a1 + 160) = &unk_4A15B10;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1478;
  *(_QWORD *)(a1 + 592) = sub_239E4F0;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v10 = a3;
  v37 = a1 + 176;
  v11 = *v10;
  v12 = **v10;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v12;
  *(_DWORD *)(a1 + 152) = *v11;
  v13 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v14 = a5[1];
  *(_QWORD *)(a1 + 40) = v13;
  v15 = *((unsigned int *)a6 + 2);
  *(_QWORD *)(a1 + 48) = v14;
  v16 = *a6;
  v39 = *a6 + 40 * v15;
  if ( *a6 != v39 )
  {
    do
    {
      v17 = *(_DWORD *)(v16 + 16);
      v18 = *(_QWORD *)(v16 + 24);
      v40[4] = &unk_4A15AF0;
      v19 = (const __m128i *)v40;
      v20 = *(_QWORD *)(v16 + 32);
      v21 = *(_QWORD *)v16;
      v42 = 1;
      v22 = *(_QWORD *)(v16 + 8);
      v23 = *(unsigned int *)(a1 + 188);
      v41 = v17;
      v24 = *(unsigned int *)(a1 + 184);
      v40[2] = v18;
      v40[3] = v20;
      v25 = *(_QWORD *)(a1 + 176);
      v26 = v24 + 1;
      v40[0] = v21;
      v27 = v24;
      v40[1] = v22;
      if ( v24 + 1 > v23 )
      {
        if ( v25 > (unsigned __int64)v40 )
        {
          v34 = v22;
          v35 = v21;
LABEL_11:
          sub_23ADBB0(v37, v26, v24, v25, v22, v21);
          v24 = *(unsigned int *)(a1 + 184);
          v25 = *(_QWORD *)(a1 + 176);
          v21 = v35;
          v22 = v34;
          v27 = *(_DWORD *)(a1 + 184);
          goto LABEL_3;
        }
        v34 = v22;
        v35 = v21;
        v24 = v25 + 48 * v24;
        if ( (unsigned __int64)v40 >= v24 )
          goto LABEL_11;
        v33 = (char *)v40 - v25;
        sub_23ADBB0(v37, v26, v24, v25, v22, v21);
        v25 = *(_QWORD *)(a1 + 176);
        v24 = *(unsigned int *)(a1 + 184);
        v22 = v34;
        v21 = v35;
        v19 = (const __m128i *)&v33[v25];
        v27 = *(_DWORD *)(a1 + 184);
      }
LABEL_3:
      v28 = (__m128i *)(v25 + 48 * v24);
      if ( v28 )
      {
        v29 = _mm_loadu_si128(v19 + 1);
        *v28 = _mm_loadu_si128(v19);
        v28[1] = v29;
        v28[2].m128i_i32[2] = v19[2].m128i_i32[2];
        v30 = v19[2].m128i_i8[12];
        v28[2].m128i_i64[0] = (__int64)&unk_4A15AF0;
        v28[2].m128i_i8[12] = v30;
        v27 = *(_DWORD *)(a1 + 184);
      }
      v31 = *(_QWORD *)(a1 + 168);
      v16 += 40;
      *(_DWORD *)(a1 + 184) = v27 + 1;
      sub_C52F90(v31, v21, v22);
    }
    while ( v39 != v16 );
  }
  return sub_C53130(a1);
}
