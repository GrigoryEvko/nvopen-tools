// Function: sub_26DF3F0
// Address: 0x26df3f0
//
unsigned __int64 __fastcall sub_26DF3F0(__int64 a1, const char *a2, int **a3, __int64 *a4, __int64 *a5, _BYTE *a6)
{
  __int64 v6; // r15
  size_t v9; // rax
  __int64 v10; // r8
  int *v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // r15
  int v16; // edx
  __int64 v17; // rsi
  const __m128i *v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // r10
  __int64 v21; // r9
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r11
  int v26; // esi
  __m128i *v27; // rdx
  __m128i v28; // xmm1
  __int8 v29; // cl
  __int64 v30; // rdi
  __int64 v31; // rdx
  char *v33; // rbx
  __int64 v34; // [rsp+0h] [rbp-90h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+20h] [rbp-70h]
  __int64 v39; // [rsp+28h] [rbp-68h]
  _QWORD v40[5]; // [rsp+30h] [rbp-60h] BYREF
  int v41; // [rsp+58h] [rbp-38h]
  char v42; // [rsp+5Ch] [rbp-34h]

  v6 = a1;
  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 156) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A1F420;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)a1 = &unk_4A1F490;
  *(_QWORD *)(a1 + 160) = &unk_4A1F440;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1538;
  *(_QWORD *)(a1 + 592) = sub_260FC60;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v11 = *a3;
  v38 = a1 + 176;
  v12 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v12;
  v13 = *((unsigned int *)a4 + 2);
  *(_DWORD *)(a1 + 152) = *v11;
  v39 = *a4 + 40 * v13;
  if ( *a4 != v39 )
  {
    v14 = *a4;
    while ( 1 )
    {
      v16 = *(_DWORD *)(v14 + 16);
      v17 = *(_QWORD *)(v14 + 24);
      v40[4] = &unk_4A1F420;
      v18 = (const __m128i *)v40;
      v19 = *(_QWORD *)(v14 + 32);
      v20 = *(_QWORD *)v14;
      v42 = 1;
      v21 = *(_QWORD *)(v14 + 8);
      v22 = *(unsigned int *)(a1 + 188);
      v41 = v16;
      v23 = *(unsigned int *)(a1 + 184);
      v40[2] = v17;
      v40[3] = v19;
      v24 = *(_QWORD *)(a1 + 176);
      v25 = v23 + 1;
      v40[0] = v20;
      v26 = v23;
      v40[1] = v21;
      if ( v23 + 1 > v22 )
      {
        if ( v24 > (unsigned __int64)v40 )
        {
          v34 = v21;
          v35 = v20;
LABEL_13:
          sub_26166F0(v38, v25, v23, v24, v10, v21);
          v23 = *(unsigned int *)(a1 + 184);
          v24 = *(_QWORD *)(a1 + 176);
          v20 = v35;
          v21 = v34;
          v26 = *(_DWORD *)(a1 + 184);
          goto LABEL_4;
        }
        v34 = v21;
        v35 = v20;
        v23 = v24 + 48 * v23;
        if ( (unsigned __int64)v40 >= v23 )
          goto LABEL_13;
        v33 = (char *)v40 - v24;
        sub_26166F0(v38, v25, v23, v24, v10, v21);
        v24 = *(_QWORD *)(a1 + 176);
        v23 = *(unsigned int *)(a1 + 184);
        v21 = v34;
        v20 = v35;
        v18 = (const __m128i *)&v33[v24];
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
        v27[2].m128i_i64[0] = (__int64)&unk_4A1F420;
        v27[2].m128i_i8[12] = v29;
        v26 = *(_DWORD *)(a1 + 184);
      }
      v30 = *(_QWORD *)(a1 + 168);
      v14 += 40;
      *(_DWORD *)(a1 + 184) = v26 + 1;
      sub_C52F90(v30, v20, v21);
      if ( v39 == v14 )
      {
        v6 = a1;
        break;
      }
    }
  }
  v31 = *a5;
  *(_QWORD *)(v6 + 48) = a5[1];
  *(_QWORD *)(v6 + 40) = v31;
  *(_BYTE *)(v6 + 12) = (32 * (*a6 & 3)) | *(_BYTE *)(v6 + 12) & 0x9F;
  return sub_C53130(v6);
}
