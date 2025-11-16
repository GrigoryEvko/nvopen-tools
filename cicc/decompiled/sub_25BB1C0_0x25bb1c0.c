// Function: sub_25BB1C0
// Address: 0x25bb1c0
//
_QWORD *__fastcall sub_25BB1C0(__int64 a1, const char *a2, _QWORD *a3, int **a4, __int64 a5)
{
  size_t v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  int *v12; // rax
  int v13; // edx
  _QWORD *result; // rax
  _QWORD *v15; // r15
  int v16; // edx
  __int64 v17; // rsi
  const __m128i *v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // r10
  __int64 v21; // r8
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r11
  int v26; // esi
  __m128i *v27; // rdx
  __m128i v28; // xmm1
  __int8 v29; // cl
  __int64 v30; // rdi
  char *v31; // rbx
  __int64 v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  _QWORD v36[5]; // [rsp+20h] [rbp-60h] BYREF
  int v37; // [rsp+48h] [rbp-38h]
  char v38; // [rsp+4Ch] [rbp-34h]

  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v11 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  *(_QWORD *)(a1 + 48) = v11;
  v12 = *a4;
  v13 = **a4;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v13;
  *(_DWORD *)(a1 + 152) = *v12;
  result = *(_QWORD **)a5;
  v35 = *(_QWORD *)a5 + 40LL * *(unsigned int *)(a5 + 8);
  if ( *(_QWORD *)a5 != v35 )
  {
    v15 = *(_QWORD **)a5;
    v34 = a1 + 176;
    do
    {
      v16 = *((_DWORD *)v15 + 4);
      v17 = v15[3];
      v36[4] = &unk_4A1F110;
      v18 = (const __m128i *)v36;
      v19 = v15[4];
      v20 = *v15;
      v38 = 1;
      v21 = v15[1];
      v22 = *(unsigned int *)(a1 + 188);
      v37 = v16;
      v23 = *(unsigned int *)(a1 + 184);
      v36[2] = v17;
      v36[3] = v19;
      v24 = *(_QWORD *)(a1 + 176);
      v25 = v23 + 1;
      v36[0] = v20;
      v26 = v23;
      v36[1] = v21;
      if ( v23 + 1 > v22 )
      {
        if ( v24 > (unsigned __int64)v36 )
        {
          v32 = v21;
          v33 = v20;
LABEL_12:
          sub_25BB100(v34, v25, v23, v24, v21, v10);
          v23 = *(unsigned int *)(a1 + 184);
          v24 = *(_QWORD *)(a1 + 176);
          v20 = v33;
          v21 = v32;
          v26 = *(_DWORD *)(a1 + 184);
          goto LABEL_4;
        }
        v32 = v21;
        v33 = v20;
        v23 = v24 + 48 * v23;
        if ( (unsigned __int64)v36 >= v23 )
          goto LABEL_12;
        v31 = (char *)v36 - v24;
        sub_25BB100(v34, v25, v23, v24, v21, v10);
        v24 = *(_QWORD *)(a1 + 176);
        v23 = *(unsigned int *)(a1 + 184);
        v21 = v32;
        v20 = v33;
        v18 = (const __m128i *)&v31[v24];
        v26 = *(_DWORD *)(a1 + 184);
      }
LABEL_4:
      v27 = (__m128i *)(v24 + 48 * v23);
      if ( v27 )
      {
        v28 = _mm_loadu_si128(v18 + 1);
        *v27 = _mm_loadu_si128(v18);
        v27[1] = v28;
        v27[2].m128i_i32[2] = v18[2].m128i_i32[2];
        v29 = v18[2].m128i_i8[12];
        v27[2].m128i_i64[0] = (__int64)&unk_4A1F110;
        v27[2].m128i_i8[12] = v29;
        v26 = *(_DWORD *)(a1 + 184);
      }
      v30 = *(_QWORD *)(a1 + 168);
      v15 += 5;
      *(_DWORD *)(a1 + 184) = v26 + 1;
      result = sub_C52F90(v30, v20, v21);
    }
    while ( (_QWORD *)v35 != v15 );
  }
  return result;
}
