// Function: sub_3746D50
// Address: 0x3746d50
//
__int64 __fastcall sub_3746D50(__int64 *a1, __int64 a2, unsigned __int8 *a3, unsigned int a4, __int64 a5, __int64 a6)
{
  int v9; // edx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  const __m128i *v21; // r13
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r11
  __m128i *v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 *v27; // rsi
  __int64 v28; // rcx
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // r10
  const __m128i *v31; // r13
  unsigned __int64 v32; // rdx
  __m128i *v33; // rax
  char *v34; // r10
  char v35; // al
  __int64 result; // rax
  __int64 v37; // rax
  const __m128i *v38; // r13
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // r10
  unsigned __int64 v41; // rdx
  __m128i *v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rcx
  const __m128i *v45; // r13
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // r10
  __m128i *v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // rcx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  const __m128i *v54; // rdx
  __m128i *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rax
  unsigned int v59; // ecx
  __int64 *v60; // rdx
  __int64 v61; // r11
  int v62; // eax
  unsigned __int64 v63; // rcx
  const void *v64; // rsi
  char *v65; // r13
  const void *v66; // rsi
  char *v67; // r13
  const void *v68; // rsi
  char *v69; // r13
  int v70; // edx
  unsigned __int64 v71; // r13
  const void *v72; // rsi
  const void *v73; // rsi
  char *v74; // r13
  int v75; // r13d
  char *v76; // [rsp+8h] [rbp-78h]
  int v77; // [rsp+1Ch] [rbp-64h]
  __int64 v78; // [rsp+20h] [rbp-60h] BYREF
  int v79; // [rsp+28h] [rbp-58h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  __int64 v82; // [rsp+40h] [rbp-40h]

  v9 = *a3;
  if ( v9 == 40 )
  {
    v10 = 32LL * (unsigned int)sub_B491D0((__int64)a3);
  }
  else
  {
    v10 = 0;
    if ( v9 != 85 )
    {
      v10 = 64;
      if ( v9 != 34 )
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_13;
  v11 = sub_BD2BC0((__int64)a3);
  v13 = v11 + v12;
  if ( (a3[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v13 >> 4) )
LABEL_71:
      BUG();
LABEL_13:
    v17 = 0;
    goto LABEL_14;
  }
  if ( !(unsigned int)((v13 - sub_BD2BC0((__int64)a3)) >> 4) )
    goto LABEL_13;
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_71;
  v14 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
  if ( (a3[7] & 0x80u) == 0 )
    BUG();
  v15 = sub_BD2BC0((__int64)a3);
  v17 = 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
LABEL_14:
  v18 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  v77 = (32 * v18 - 32 - v10 - v17) >> 5;
  if ( a4 == v77 )
    return 1;
  while ( 1 )
  {
    v34 = *(char **)&a3[32 * (a4 - v18)];
    v35 = *v34;
    if ( *v34 == 17 )
    {
      v19 = *(unsigned int *)(a2 + 8);
      v20 = *(unsigned int *)(a2 + 12);
      v78 = 1;
      v21 = (const __m128i *)&v78;
      v80 = 0;
      v22 = *(_QWORD *)a2;
      v23 = v19 + 1;
      v81 = 2;
      if ( v19 + 1 <= v20 )
        goto LABEL_17;
      v64 = (const void *)(a2 + 16);
      if ( v22 > (unsigned __int64)&v78 )
      {
        v76 = v34;
      }
      else
      {
        v76 = v34;
        if ( (unsigned __int64)&v78 < v22 + 40 * v19 )
        {
          v65 = (char *)&v78 - v22;
          sub_C8D5F0(a2, v64, v23, 0x28u, a5, a6);
          v22 = *(_QWORD *)a2;
          v19 = *(unsigned int *)(a2 + 8);
          v34 = v76;
          v21 = (const __m128i *)&v65[*(_QWORD *)a2];
          goto LABEL_17;
        }
      }
      sub_C8D5F0(a2, v64, v23, 0x28u, a5, a6);
      v22 = *(_QWORD *)a2;
      v19 = *(unsigned int *)(a2 + 8);
      v34 = v76;
LABEL_17:
      v24 = (__m128i *)(v22 + 40 * v19);
      *v24 = _mm_loadu_si128(v21);
      v24[1] = _mm_loadu_si128(v21 + 1);
      v24[2].m128i_i64[0] = v21[2].m128i_i64[0];
      v25 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v25;
      v26 = *((_DWORD *)v34 + 8);
      v27 = (__int64 *)*((_QWORD *)v34 + 3);
      if ( v26 > 0x40 )
      {
        v28 = *v27;
      }
      else
      {
        v28 = 0;
        if ( v26 )
          v28 = (__int64)((_QWORD)v27 << (64 - (unsigned __int8)v26)) >> (64 - (unsigned __int8)v26);
      }
      v81 = v28;
      v29 = *(unsigned int *)(a2 + 12);
      v30 = v25 + 1;
      v31 = (const __m128i *)&v78;
      v78 = 1;
      v32 = *(_QWORD *)a2;
      v80 = 0;
      if ( v25 + 1 <= v29 )
        goto LABEL_21;
      v66 = (const void *)(a2 + 16);
      if ( v32 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v32 + 40 * v25 )
        goto LABEL_68;
      goto LABEL_49;
    }
    if ( v35 == 20 )
    {
      v43 = *(unsigned int *)(a2 + 8);
      v44 = *(unsigned int *)(a2 + 12);
      v78 = 1;
      v45 = (const __m128i *)&v78;
      v80 = 0;
      v46 = *(_QWORD *)a2;
      v47 = v43 + 1;
      v81 = 2;
      if ( v43 + 1 > v44 )
      {
        v73 = (const void *)(a2 + 16);
        if ( v46 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v46 + 40 * v43 )
        {
          v45 = (const __m128i *)&v78;
          sub_C8D5F0(a2, v73, v47, 0x28u, a5, a6);
          v46 = *(_QWORD *)a2;
          v43 = *(unsigned int *)(a2 + 8);
        }
        else
        {
          v74 = (char *)&v78 - v46;
          sub_C8D5F0(a2, v73, v47, 0x28u, a5, a6);
          v46 = *(_QWORD *)a2;
          v43 = *(unsigned int *)(a2 + 8);
          v45 = (const __m128i *)&v74[*(_QWORD *)a2];
        }
      }
      v48 = (__m128i *)(v46 + 40 * v43);
      *v48 = _mm_loadu_si128(v45);
      v48[1] = _mm_loadu_si128(v45 + 1);
      v49 = v45[2].m128i_i64[0];
      v78 = 1;
      v48[2].m128i_i64[0] = v49;
      LODWORD(v48) = *(_DWORD *)(a2 + 8);
      v50 = *(unsigned int *)(a2 + 12);
      v80 = 0;
      v51 = (unsigned int)((_DWORD)v48 + 1);
      v81 = 0;
      v52 = v51 + 1;
      *(_DWORD *)(a2 + 8) = v51;
      if ( v51 + 1 > v50 )
      {
        v71 = *(_QWORD *)a2;
        v72 = (const void *)(a2 + 16);
        if ( *(_QWORD *)a2 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v71 + 40 * v51 )
        {
          sub_C8D5F0(a2, v72, v52, 0x28u, a5, a6);
          v53 = *(_QWORD *)a2;
          v51 = *(unsigned int *)(a2 + 8);
          v54 = (const __m128i *)&v78;
        }
        else
        {
          sub_C8D5F0(a2, v72, v52, 0x28u, a5, a6);
          v53 = *(_QWORD *)a2;
          v51 = *(unsigned int *)(a2 + 8);
          v54 = (const __m128i *)((char *)&v78 + *(_QWORD *)a2 - v71);
        }
      }
      else
      {
        v53 = *(_QWORD *)a2;
        v54 = (const __m128i *)&v78;
      }
      v55 = (__m128i *)(v53 + 40 * v51);
      *v55 = _mm_loadu_si128(v54);
      v55[1] = _mm_loadu_si128(v54 + 1);
      v55[2].m128i_i64[0] = v54[2].m128i_i64[0];
      ++*(_DWORD *)(a2 + 8);
LABEL_22:
      if ( ++a4 == v77 )
        return 1;
      goto LABEL_23;
    }
    if ( v35 == 60 )
      break;
    result = sub_3746830(a1, *(_QWORD *)&a3[32 * (a4 - v18)]);
    if ( !(_DWORD)result )
      return result;
    v79 = result;
    v37 = *(unsigned int *)(a2 + 8);
    v38 = (const __m128i *)&v78;
    v39 = *(unsigned int *)(a2 + 12);
    v78 = 0;
    v40 = v37 + 1;
    v80 = 0;
    v41 = *(_QWORD *)a2;
    v81 = 0;
    v82 = 0;
    if ( v37 + 1 > v39 )
    {
      v68 = (const void *)(a2 + 16);
      if ( v41 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v41 + 40 * v37 )
      {
        v38 = (const __m128i *)&v78;
        sub_C8D5F0(a2, v68, v40, 0x28u, a5, a6);
        v41 = *(_QWORD *)a2;
        v37 = *(unsigned int *)(a2 + 8);
      }
      else
      {
        v69 = (char *)&v78 - v41;
        sub_C8D5F0(a2, v68, v40, 0x28u, a5, a6);
        v41 = *(_QWORD *)a2;
        v37 = *(unsigned int *)(a2 + 8);
        v38 = (const __m128i *)&v69[*(_QWORD *)a2];
      }
    }
    ++a4;
    v42 = (__m128i *)(v41 + 40 * v37);
    *v42 = _mm_loadu_si128(v38);
    v42[1] = _mm_loadu_si128(v38 + 1);
    v42[2].m128i_i64[0] = v38[2].m128i_i64[0];
    ++*(_DWORD *)(a2 + 8);
    if ( a4 == v77 )
      return 1;
LABEL_23:
    v18 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  }
  v56 = a1[5];
  v57 = *(_QWORD *)(v56 + 256);
  v58 = *(unsigned int *)(v56 + 272);
  if ( !(_DWORD)v58 )
    return 0;
  v59 = (v58 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
  v60 = (__int64 *)(v57 + 16LL * v59);
  v61 = *v60;
  if ( v34 == (char *)*v60 )
  {
LABEL_39:
    if ( v60 == (__int64 *)(v57 + 16 * v58) )
      return 0;
    v62 = *((_DWORD *)v60 + 2);
    v63 = *(unsigned int *)(a2 + 12);
    v31 = (const __m128i *)&v78;
    v78 = 5;
    v80 = 0;
    v32 = *(_QWORD *)a2;
    LODWORD(v81) = v62;
    v25 = *(unsigned int *)(a2 + 8);
    v30 = v25 + 1;
    if ( v25 + 1 <= v63 )
    {
LABEL_21:
      v33 = (__m128i *)(v32 + 40 * v25);
      *v33 = _mm_loadu_si128(v31);
      v33[1] = _mm_loadu_si128(v31 + 1);
      v33[2].m128i_i64[0] = v31[2].m128i_i64[0];
      ++*(_DWORD *)(a2 + 8);
      goto LABEL_22;
    }
    v66 = (const void *)(a2 + 16);
    if ( v32 > (unsigned __int64)&v78 || (unsigned __int64)&v78 >= v32 + 40 * v25 )
    {
LABEL_68:
      v31 = (const __m128i *)&v78;
      sub_C8D5F0(a2, v66, v30, 0x28u, a5, a6);
      v32 = *(_QWORD *)a2;
      v25 = *(unsigned int *)(a2 + 8);
      goto LABEL_21;
    }
LABEL_49:
    v67 = (char *)&v78 - v32;
    sub_C8D5F0(a2, v66, v30, 0x28u, a5, a6);
    v32 = *(_QWORD *)a2;
    v25 = *(unsigned int *)(a2 + 8);
    v31 = (const __m128i *)&v67[*(_QWORD *)a2];
    goto LABEL_21;
  }
  v70 = 1;
  while ( v61 != -4096 )
  {
    v75 = v70 + 1;
    v59 = (v58 - 1) & (v70 + v59);
    v60 = (__int64 *)(v57 + 16LL * v59);
    v61 = *v60;
    if ( v34 == (char *)*v60 )
      goto LABEL_39;
    v70 = v75;
  }
  return 0;
}
