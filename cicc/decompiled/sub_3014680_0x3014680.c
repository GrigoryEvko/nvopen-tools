// Function: sub_3014680
// Address: 0x3014680
//
__int64 __fastcall sub_3014680(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r14
  unsigned __int8 *v12; // rax
  bool v13; // zf
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  const __m128i *v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // r9
  __m128i *v19; // rax
  unsigned int v20; // r15d
  __int64 result; // rax
  __int64 v22; // r14
  unsigned __int8 *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r13
  unsigned __int8 *v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // rsi
  int v31; // ecx
  __int64 v32; // rdx
  int v33; // edi
  __int64 v34; // rdi
  const void *v35; // rsi
  char *v36; // r15
  unsigned __int64 v37; // rsi
  const __m128i *v38; // rdx
  unsigned __int64 v39; // rcx
  __int64 v40; // rax
  unsigned __int64 v41; // r9
  __m128i *v42; // rax
  unsigned int v43; // r14d
  __int64 v44; // r13
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rsi
  unsigned int v48; // ecx
  __int64 v49; // rdi
  const void *v50; // rsi
  char *v51; // r13
  __int64 v52; // [rsp+10h] [rbp-60h]
  __int64 v54; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int8 *v55; // [rsp+28h] [rbp-48h]
  unsigned __int64 v56; // [rsp+30h] [rbp-40h]

  v52 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)a2 == 39 )
  {
    v5 = *(_QWORD *)(a2 - 8);
    v6 = (__int64 *)(v5 + 32);
    v7 = (__int64 *)(v5 + 64);
    if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
      v7 = v6;
    v8 = sub_AA4FF0(*v7);
    v9 = v8;
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)(v8 + 16);
    v11 = v8 - 24;
    v12 = sub_BD3990(*(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) - 24), a2);
    v13 = *v12 == 0;
    BYTE4(v54) = 0;
    if ( !v13 )
      v12 = 0;
    LODWORD(v54) = a3;
    v14 = *(unsigned int *)(a1 + 524);
    v56 = v10 & 0xFFFFFFFFFFFFFFFBLL;
    v15 = *(_QWORD *)(a1 + 512);
    v16 = (const __m128i *)&v54;
    v55 = v12;
    v17 = *(unsigned int *)(a1 + 520);
    v18 = v17 + 1;
    if ( v17 + 1 > v14 )
    {
      v34 = a1 + 512;
      v35 = (const void *)(a1 + 528);
      if ( v15 > (unsigned __int64)&v54 || (unsigned __int64)&v54 >= v15 + 24 * v17 )
      {
        sub_C8D5F0(v34, v35, v18, 0x18u, (__int64)&v54, v18);
        v15 = *(_QWORD *)(a1 + 512);
        v17 = *(unsigned int *)(a1 + 520);
      }
      else
      {
        v36 = (char *)&v54 - v15;
        sub_C8D5F0(v34, v35, v18, 0x18u, (__int64)&v54, v18);
        v15 = *(_QWORD *)(a1 + 512);
        v17 = *(unsigned int *)(a1 + 520);
        v16 = (const __m128i *)&v36[v15];
      }
    }
    v19 = (__m128i *)(v15 + 24 * v17);
    *v19 = _mm_loadu_si128(v16);
    v19[1].m128i_i64[0] = v16[1].m128i_i64[0];
    v20 = *(_DWORD *)(a1 + 520);
    v54 = a2;
    *(_DWORD *)(a1 + 520) = v20 + 1;
    *(_DWORD *)sub_3014430(a1, &v54) = v20;
    v54 = v11;
    *(_DWORD *)sub_3014430(a1, &v54) = v20;
    result = v52;
    v22 = *(_QWORD *)(v52 + 16);
    if ( v22 )
    {
      while ( 1 )
      {
        v23 = *(unsigned __int8 **)(v22 + 24);
        result = (unsigned int)*v23 - 30;
        if ( (unsigned __int8)(*v23 - 30) <= 0xAu )
          break;
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
          goto LABEL_17;
      }
LABEL_12:
      result = sub_3012100(*((_QWORD *)v23 + 5), **(_QWORD **)(a2 - 8));
      if ( result )
      {
        v24 = sub_AA4FF0(result);
        if ( v24 )
          v24 -= 24;
        result = sub_3014680(a1, v24, v20);
      }
      while ( 1 )
      {
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
          break;
        v23 = *(unsigned __int8 **)(v22 + 24);
        result = (unsigned int)*v23 - 30;
        if ( (unsigned __int8)(*v23 - 30) <= 0xAu )
          goto LABEL_12;
      }
    }
LABEL_17:
    v25 = *(_QWORD *)(v9 - 8);
    if ( v25 )
    {
      while ( 1 )
      {
        v26 = *(unsigned __int8 **)(v25 + 24);
        result = *v26;
        if ( (_BYTE)result != 39 )
          goto LABEL_25;
        if ( (v26[2] & 1) == 0 )
          break;
        result = *(_QWORD *)(*((_QWORD *)v26 - 1) + 32LL);
        if ( !result )
          break;
        if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
        {
          v27 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
          if ( result == v27 )
          {
            if ( v27 )
              break;
          }
        }
LABEL_31:
        v25 = *(_QWORD *)(v25 + 8);
        if ( !v25 )
          return result;
      }
      sub_3014680(a1, *(_QWORD *)(v25 + 24), a3);
      result = *v26;
LABEL_25:
      if ( (_BYTE)result == 80 )
      {
        result = sub_3011DA0((__int64)v26);
        if ( !result
          || (*(_BYTE *)(a2 + 2) & 1) != 0 && (v28 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL), result == v28) && v28 )
        {
          result = sub_3014680(a1, v26, a3);
        }
      }
      goto LABEL_31;
    }
  }
  else
  {
    v29 = *(_DWORD *)(a1 + 24);
    v30 = *(_QWORD *)(a1 + 8);
    if ( v29 )
    {
      v31 = v29 - 1;
      result = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = *(_QWORD *)(v30 + 16 * result);
      if ( a2 == v32 )
        return result;
      v33 = 1;
      while ( v32 != -4096 )
      {
        result = v31 & (unsigned int)(v33 + result);
        v32 = *(_QWORD *)(v30 + 16LL * (unsigned int)result);
        if ( a2 == v32 )
          return result;
        ++v33;
      }
    }
    BYTE4(v54) = 1;
    v37 = *(unsigned int *)(a1 + 524);
    v55 = 0;
    v38 = (const __m128i *)&v54;
    LODWORD(v54) = a3;
    v39 = *(_QWORD *)(a1 + 512);
    v56 = v52 & 0xFFFFFFFFFFFFFFFBLL;
    v40 = *(unsigned int *)(a1 + 520);
    v41 = v40 + 1;
    if ( v40 + 1 > v37 )
    {
      v49 = a1 + 512;
      v50 = (const void *)(a1 + 528);
      if ( v39 > (unsigned __int64)&v54 || (unsigned __int64)&v54 >= v39 + 24 * v40 )
      {
        sub_C8D5F0(v49, v50, v41, 0x18u, (__int64)&v54, v41);
        v39 = *(_QWORD *)(a1 + 512);
        v40 = *(unsigned int *)(a1 + 520);
        v38 = (const __m128i *)&v54;
      }
      else
      {
        v51 = (char *)&v54 - v39;
        sub_C8D5F0(v49, v50, v41, 0x18u, (__int64)&v54, v41);
        v39 = *(_QWORD *)(a1 + 512);
        v40 = *(unsigned int *)(a1 + 520);
        v38 = (const __m128i *)&v51[v39];
      }
    }
    v42 = (__m128i *)(v39 + 24 * v40);
    *v42 = _mm_loadu_si128(v38);
    v42[1].m128i_i64[0] = v38[1].m128i_i64[0];
    v43 = *(_DWORD *)(a1 + 520);
    v54 = a2;
    *(_DWORD *)(a1 + 520) = v43 + 1;
    *(_DWORD *)sub_3014430(a1, &v54) = v43;
    v44 = *(_QWORD *)(v52 + 16);
    if ( v44 )
    {
      while ( 1 )
      {
        v45 = *(_QWORD *)(v44 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v45 - 30) <= 0xAu )
          break;
        v44 = *(_QWORD *)(v44 + 8);
        if ( !v44 )
          goto LABEL_55;
      }
LABEL_50:
      v46 = sub_3012100(*(_QWORD *)(v45 + 40), *(_QWORD *)(a2 - 32));
      if ( v46 )
      {
        v47 = sub_AA4FF0(v46);
        if ( v47 )
          v47 -= 24;
        sub_3014680(a1, v47, v43);
      }
      while ( 1 )
      {
        v44 = *(_QWORD *)(v44 + 8);
        if ( !v44 )
          break;
        v45 = *(_QWORD *)(v44 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v45 - 30) <= 0xAu )
          goto LABEL_50;
      }
    }
LABEL_55:
    for ( result = *(_QWORD *)(a2 + 16); result; result = *(_QWORD *)(result + 8) )
    {
      v48 = **(unsigned __int8 **)(result + 24) - 39;
      if ( v48 <= 0x38 && ((1LL << v48) & 0x100060000000001LL) != 0 )
        sub_C64ED0("Cleanup funclets for the SEH personality cannot contain exceptional actions", 1u);
    }
  }
  return result;
}
