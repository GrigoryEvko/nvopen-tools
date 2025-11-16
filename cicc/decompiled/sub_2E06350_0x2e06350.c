// Function: sub_2E06350
// Address: 0x2e06350
//
__int64 __fastcall sub_2E06350(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 v6; // rax
  unsigned int v7; // r13d
  int v8; // esi
  __int64 v9; // r8
  __m128i *v10; // r9
  int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rcx
  __m128i *v16; // rdx
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v22; // rax
  unsigned int v23; // r13d
  int v24; // eax
  int v25; // r11d
  _QWORD *v26; // r10
  unsigned int i; // ecx
  __m128i *v28; // rdx
  __int8 v29; // si
  char v30; // al
  int v31; // eax
  char v32; // al
  __int64 v33; // r9
  unsigned __int64 v34; // rsi
  __m128i *v35; // rdi
  __m128i v36; // xmm6
  __m128i v37; // xmm7
  __int64 v38; // rax
  __m128i *v39; // rax
  char v40; // al
  char v41; // al
  unsigned int v42; // ecx
  __int8 *v43; // r12
  _QWORD *v44; // [rsp+0h] [rbp-120h]
  int v45; // [rsp+0h] [rbp-120h]
  _QWORD *v46; // [rsp+0h] [rbp-120h]
  int v47; // [rsp+8h] [rbp-118h]
  __m128i *v48; // [rsp+8h] [rbp-118h]
  int v49; // [rsp+8h] [rbp-118h]
  __m128i *v50; // [rsp+10h] [rbp-110h]
  unsigned int v51; // [rsp+10h] [rbp-110h]
  __m128i *v52; // [rsp+10h] [rbp-110h]
  unsigned int v53; // [rsp+18h] [rbp-108h]
  _QWORD *v54; // [rsp+18h] [rbp-108h]
  unsigned int v55; // [rsp+18h] [rbp-108h]
  __m128i *v56; // [rsp+20h] [rbp-100h]
  __m128i *v57; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+28h] [rbp-F8h]
  _QWORD v59[6]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD v60[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+70h] [rbp-B0h]
  __m128i v62; // [rsp+90h] [rbp-90h]
  __m128i v63; // [rsp+A0h] [rbp-80h]
  __int64 v64; // [rsp+B0h] [rbp-70h]
  __m128i v65; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v66; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v67; // [rsp+E0h] [rbp-40h]
  __int64 v68; // [rsp+E8h] [rbp-38h]

  LODWORD(v68) = 0;
  v6 = a2[2].m128i_i64[0];
  v7 = *(_DWORD *)(a1 + 24);
  v62 = _mm_loadu_si128(a2);
  v64 = v6;
  v67 = v6;
  v63 = _mm_loadu_si128(a2 + 1);
  v65 = v62;
  v66 = v63;
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    v59[0] = 0;
LABEL_3:
    v8 = 2 * v7;
    goto LABEL_4;
  }
  v22 = *(_QWORD *)(a1 + 8);
  v59[0] = 21;
  v23 = v7 - 1;
  v59[2] = 0;
  v58 = v22;
  v60[0] = 22;
  v61 = 0;
  v24 = sub_2EAE040(&v65);
  v25 = 1;
  v10 = 0;
  v26 = v59;
  for ( i = v23 & v24; ; i = v23 & v42 )
  {
    v28 = (__m128i *)(v58 + 48LL * i);
    if ( (unsigned __int8)(v65.m128i_i8[0] - 21) > 1u )
    {
      v44 = v26;
      v47 = v25;
      v50 = v10;
      v53 = i;
      v56 = (__m128i *)(v58 + 48LL * i);
      v30 = sub_2EAB6C0(&v65, v56);
      v28 = v56;
      i = v53;
      v10 = v50;
      v25 = v47;
      v26 = v44;
      if ( v30 )
        return *(_QWORD *)(a1 + 32) + 48LL * v28[2].m128i_u32[2];
      v29 = v56->m128i_i8[0];
    }
    else
    {
      v29 = v28->m128i_i8[0];
      if ( v65.m128i_i8[0] == v28->m128i_i8[0] )
        return *(_QWORD *)(a1 + 32) + 48LL * v28[2].m128i_u32[2];
    }
    if ( (unsigned __int8)(v29 - 21) <= 1u )
    {
      if ( v29 == LOBYTE(v59[0]) )
        break;
LABEL_36:
      if ( LOBYTE(v60[0]) != v29 )
        goto LABEL_35;
      goto LABEL_33;
    }
    v45 = v25;
    v48 = v10;
    v51 = i;
    v54 = v26;
    v57 = v28;
    v40 = sub_2EAB6C0(v28, v26);
    v28 = v57;
    v26 = v54;
    i = v51;
    v10 = v48;
    v25 = v45;
    if ( v40 )
      break;
    v29 = v57->m128i_i8[0];
    if ( (unsigned __int8)(v57->m128i_i8[0] - 21) <= 1u )
      goto LABEL_36;
    v46 = v54;
    v49 = v25;
    v52 = v10;
    v55 = i;
    v41 = sub_2EAB6C0(v57, v60);
    v28 = v57;
    i = v55;
    v10 = v52;
    v25 = v49;
    v26 = v46;
    if ( !v41 )
      goto LABEL_35;
LABEL_33:
    if ( !v10 )
      v10 = v28;
LABEL_35:
    v42 = v25 + i;
    ++v25;
  }
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
    v10 = v28;
  v31 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v59[0] = v10;
  v11 = v31 + 1;
  if ( 4 * v11 >= 3 * v7 )
    goto LABEL_3;
  if ( v7 - (v11 + *(_DWORD *)(a1 + 20)) <= v7 >> 3 )
  {
    v8 = v7;
LABEL_4:
    sub_2E05F00(a1, v8);
    sub_2E01450(a1, &v65, v59);
    v10 = (__m128i *)v59[0];
    v11 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v11;
  v60[0] = 21;
  v61 = 0;
  if ( (unsigned __int8)(v10->m128i_i8[0] - 21) > 1u )
  {
    v32 = sub_2EAB6C0(v10, v60);
    v10 = (__m128i *)v59[0];
    if ( !v32 )
LABEL_7:
      --*(_DWORD *)(a1 + 20);
  }
  else if ( v10->m128i_i8[0] != 21 )
  {
    goto LABEL_7;
  }
  *v10 = _mm_loadu_si128(&v65);
  v10[1] = _mm_loadu_si128(&v66);
  v10[2].m128i_i64[0] = v67;
  v10[2].m128i_i32[2] = v68;
  v10[2].m128i_i32[2] = *(_DWORD *)(a1 + 40);
  v12 = *(unsigned int *)(a1 + 40);
  v13 = *(unsigned int *)(a1 + 44);
  v14 = *(_DWORD *)(a1 + 40);
  if ( v12 >= v13 )
  {
    v33 = v12 + 1;
    v34 = *(_QWORD *)(a1 + 32);
    v35 = &v65;
    v36 = _mm_loadu_si128(a2);
    v37 = _mm_loadu_si128(a2 + 1);
    v67 = a2[2].m128i_i64[0];
    v38 = *a3;
    v65 = v36;
    v68 = v38;
    v66 = v37;
    if ( v13 < v12 + 1 )
    {
      if ( v34 > (unsigned __int64)&v65 || (unsigned __int64)&v65 >= v34 + 48 * v12 )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v12 + 1, 0x30u, v9, v33);
        v34 = *(_QWORD *)(a1 + 32);
        v12 = *(unsigned int *)(a1 + 40);
        v35 = &v65;
      }
      else
      {
        v43 = &v65.m128i_i8[-v34];
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v12 + 1, 0x30u, v9, v33);
        v34 = *(_QWORD *)(a1 + 32);
        v12 = *(unsigned int *)(a1 + 40);
        v35 = (__m128i *)&v43[v34];
      }
    }
    v39 = (__m128i *)(v34 + 48 * v12);
    *v39 = _mm_loadu_si128(v35);
    v39[1] = _mm_loadu_si128(v35 + 1);
    v39[2] = _mm_loadu_si128(v35 + 2);
    v15 = *(_QWORD *)(a1 + 32);
    v20 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v20;
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 32);
    v16 = (__m128i *)(v15 + 48 * v12);
    if ( v16 )
    {
      v17 = _mm_loadu_si128(a2);
      v18 = _mm_loadu_si128(a2 + 1);
      v16[2].m128i_i64[0] = a2[2].m128i_i64[0];
      v19 = *a3;
      *v16 = v17;
      v16[2].m128i_i64[1] = v19;
      v16[1] = v18;
      v14 = *(_DWORD *)(a1 + 40);
      v15 = *(_QWORD *)(a1 + 32);
    }
    v20 = (unsigned int)(v14 + 1);
    *(_DWORD *)(a1 + 40) = v20;
  }
  return v15 + 48 * v20 - 48;
}
