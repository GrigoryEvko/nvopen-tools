// Function: sub_36FE630
// Address: 0x36fe630
//
__int64 __fastcall sub_36FE630(__int64 a1, const char *a2, int **a3, __int64 *a4, _BYTE *a5, _QWORD *a6)
{
  __int64 v6; // r15
  size_t v9; // rax
  __int64 v10; // r8
  int *v11; // rax
  int v12; // edx
  __int64 v13; // r15
  int v15; // edx
  __int64 v16; // rsi
  const __m128i *v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // r9
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // r11
  int v25; // esi
  __m128i *v26; // rdx
  __m128i v27; // xmm1
  __int8 v28; // cl
  __int64 v29; // rdi
  __int64 result; // rax
  char *v31; // rbx
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  _QWORD v38[5]; // [rsp+30h] [rbp-60h] BYREF
  int v39; // [rsp+58h] [rbp-38h]
  char v40; // [rsp+5Ch] [rbp-34h]

  v6 = a1;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v11 = *a3;
  v12 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v12;
  *(_DWORD *)(a1 + 152) = *v11;
  v37 = *a4 + 40LL * *((unsigned int *)a4 + 2);
  if ( *a4 != v37 )
  {
    v36 = a1 + 176;
    v13 = *a4;
    while ( 1 )
    {
      v15 = *(_DWORD *)(v13 + 16);
      v16 = *(_QWORD *)(v13 + 24);
      v38[4] = &unk_4A3C490;
      v17 = (const __m128i *)v38;
      v18 = *(_QWORD *)(v13 + 32);
      v19 = *(_QWORD *)v13;
      v40 = 1;
      v20 = *(_QWORD *)(v13 + 8);
      v21 = *(unsigned int *)(a1 + 188);
      v39 = v15;
      v22 = *(unsigned int *)(a1 + 184);
      v38[2] = v16;
      v38[3] = v18;
      v23 = *(_QWORD *)(a1 + 176);
      v24 = v22 + 1;
      v38[0] = v19;
      v25 = v22;
      v38[1] = v20;
      if ( v22 + 1 > v21 )
      {
        if ( v23 > (unsigned __int64)v38 )
        {
          v32 = v20;
          v33 = v19;
LABEL_13:
          sub_36FE570(v36, v24, v22, v23, v10, v20);
          v22 = *(unsigned int *)(a1 + 184);
          v23 = *(_QWORD *)(a1 + 176);
          v19 = v33;
          v20 = v32;
          v25 = *(_DWORD *)(a1 + 184);
          goto LABEL_4;
        }
        v32 = v20;
        v33 = v19;
        v22 = v23 + 48 * v22;
        if ( (unsigned __int64)v38 >= v22 )
          goto LABEL_13;
        v31 = (char *)v38 - v23;
        sub_36FE570(v36, v24, v22, v23, v10, v20);
        v23 = *(_QWORD *)(a1 + 176);
        v22 = *(unsigned int *)(a1 + 184);
        v20 = v32;
        v19 = v33;
        v17 = (const __m128i *)&v31[v23];
        v25 = *(_DWORD *)(a1 + 184);
      }
LABEL_4:
      v26 = (__m128i *)(v23 + 48 * v22);
      if ( v26 )
      {
        v27 = _mm_loadu_si128(v17 + 1);
        *v26 = _mm_loadu_si128(v17);
        v26[1] = v27;
        v26[2].m128i_i32[2] = v17[2].m128i_i32[2];
        v28 = v17[2].m128i_i8[12];
        v26[2].m128i_i64[0] = (__int64)&unk_4A3C490;
        v26[2].m128i_i8[12] = v28;
        v25 = *(_DWORD *)(a1 + 184);
      }
      v29 = *(_QWORD *)(a1 + 168);
      v13 += 40;
      *(_DWORD *)(a1 + 184) = v25 + 1;
      sub_C52F90(v29, v19, v20);
      if ( v37 == v13 )
      {
        v6 = a1;
        break;
      }
    }
  }
  *(_BYTE *)(v6 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(v6 + 12) & 0x9F;
  result = a6[1];
  *(_QWORD *)(v6 + 40) = *a6;
  *(_QWORD *)(v6 + 48) = result;
  return result;
}
