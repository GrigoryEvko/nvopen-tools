// Function: sub_E7B400
// Address: 0xe7b400
//
__int64 __fastcall sub_E7B400(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // r9
  unsigned int v16; // esi
  __int64 v17; // rax
  __int64 v18; // r12
  int v19; // r11d
  __int64 v20; // rcx
  __int64 v21; // r15
  unsigned int v22; // edx
  __int64 result; // rax
  __int64 v24; // r10
  __int64 v25; // rdi
  __int64 v26; // rdi
  __m128i *v27; // rsi
  int v28; // eax
  int v29; // edx
  __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 *v34; // rcx
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rdi
  unsigned int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // r8
  unsigned __int64 v47; // r13
  __int64 v48; // r8
  int v49; // eax
  int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // rdi
  unsigned int v53; // r13d
  int v54; // r8d
  __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 v57; // [rsp+10h] [rbp-80h] BYREF
  __int64 v58; // [rsp+18h] [rbp-78h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+28h] [rbp-68h]
  __m128i v61; // [rsp+30h] [rbp-60h] BYREF
  __m128i v62; // [rsp+40h] [rbp-50h] BYREF
  __m128i v63; // [rsp+50h] [rbp-40h] BYREF

  v7 = a2[1];
  v61.m128i_i64[0] = a4;
  v61.m128i_i64[1] = a5;
  v63.m128i_i16[0] = 261;
  v56 = sub_E6C460(v7, (const char **)&v61);
  v11 = sub_E6C430(v7, (__int64)&v61, v8, v9, v10);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v11, 0);
  v12 = *(_QWORD *)(v7 + 1784);
  v13 = *(_QWORD *)(v7 + 1776);
  v14 = a1 + 520;
  v15 = a3;
  v62.m128i_i64[0] = v11;
  v61.m128i_i64[1] = v12;
  v16 = *(_DWORD *)(a1 + 544);
  v17 = a2[36];
  v61.m128i_i64[0] = v13;
  v62.m128i_i64[1] = v56;
  v63.m128i_i64[0] = a3;
  v63.m128i_i8[8] = 0;
  v18 = *(_QWORD *)(v17 + 8);
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 520);
    goto LABEL_29;
  }
  v19 = 1;
  v20 = *(_QWORD *)(a1 + 528);
  v21 = 0;
  v22 = (v16 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  result = v20 + 16LL * v22;
  v24 = *(_QWORD *)result;
  if ( v18 == *(_QWORD *)result )
  {
LABEL_3:
    v25 = *(unsigned int *)(result + 8);
    goto LABEL_4;
  }
  while ( v24 != -4096 )
  {
    if ( !v21 && v24 == -8192 )
      v21 = result;
    v15 = (unsigned int)(v19 + 1);
    v22 = (v16 - 1) & (v19 + v22);
    result = v20 + 16LL * v22;
    v24 = *(_QWORD *)result;
    if ( v18 == *(_QWORD *)result )
      goto LABEL_3;
    ++v19;
  }
  if ( !v21 )
    v21 = result;
  v28 = *(_DWORD *)(a1 + 536);
  ++*(_QWORD *)(a1 + 520);
  v29 = v28 + 1;
  if ( 4 * (v28 + 1) >= 3 * v16 )
  {
LABEL_29:
    sub_E7B220(v14, 2 * v16);
    v42 = *(_DWORD *)(a1 + 544);
    if ( v42 )
    {
      v30 = (unsigned int)(v42 - 1);
      v43 = *(_QWORD *)(a1 + 528);
      v44 = v30 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v29 = *(_DWORD *)(a1 + 536) + 1;
      v21 = v43 + 16LL * v44;
      v45 = *(_QWORD *)v21;
      if ( v18 != *(_QWORD *)v21 )
      {
        v15 = 1;
        v46 = 0;
        while ( v45 != -4096 )
        {
          if ( !v46 && v45 == -8192 )
            v46 = v21;
          v44 = v30 & (v15 + v44);
          v21 = v43 + 16LL * v44;
          v45 = *(_QWORD *)v21;
          if ( v18 == *(_QWORD *)v21 )
            goto LABEL_18;
          v15 = (unsigned int)(v15 + 1);
        }
        if ( v46 )
          v21 = v46;
      }
      goto LABEL_18;
    }
    goto LABEL_56;
  }
  v30 = v16 >> 3;
  if ( v16 - *(_DWORD *)(a1 + 540) - v29 <= (unsigned int)v30 )
  {
    sub_E7B220(v14, v16);
    v49 = *(_DWORD *)(a1 + 544);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a1 + 528);
      v52 = 0;
      v53 = v50 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v54 = 1;
      v29 = *(_DWORD *)(a1 + 536) + 1;
      v21 = v51 + 16LL * v53;
      v30 = *(_QWORD *)v21;
      if ( v18 != *(_QWORD *)v21 )
      {
        while ( v30 != -4096 )
        {
          if ( !v52 && v30 == -8192 )
            v52 = v21;
          v15 = (unsigned int)(v54 + 1);
          v53 = v50 & (v54 + v53);
          v21 = v51 + 16LL * v53;
          v30 = *(_QWORD *)v21;
          if ( v18 == *(_QWORD *)v21 )
            goto LABEL_18;
          ++v54;
        }
        if ( v52 )
          v21 = v52;
      }
      goto LABEL_18;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 536);
    BUG();
  }
LABEL_18:
  *(_DWORD *)(a1 + 536) = v29;
  if ( *(_QWORD *)v21 != -4096 )
    --*(_DWORD *)(a1 + 540);
  *(_QWORD *)v21 = v18;
  *(_DWORD *)(v21 + 8) = 0;
  v25 = *(unsigned int *)(a1 + 560);
  v31 = *(unsigned int *)(a1 + 564);
  v57 = v18;
  v32 = v25 + 1;
  v58 = 0;
  result = v25;
  v59 = 0;
  v60 = 0;
  if ( v25 + 1 > v31 )
  {
    v47 = *(_QWORD *)(a1 + 552);
    v48 = a1 + 552;
    if ( v47 > (unsigned __int64)&v57 || (unsigned __int64)&v57 >= v47 + 32 * v25 )
    {
      sub_E79C70(a1 + 552, v32, v31, v30, v48, v15);
      v25 = *(unsigned int *)(a1 + 560);
      v33 = *(_QWORD *)(a1 + 552);
      v34 = &v57;
      result = v25;
    }
    else
    {
      sub_E79C70(a1 + 552, v32, v31, v30, v48, v15);
      v33 = *(_QWORD *)(a1 + 552);
      v25 = *(unsigned int *)(a1 + 560);
      v34 = (__int64 *)((char *)&v57 + v33 - v47);
      result = v25;
    }
  }
  else
  {
    v33 = *(_QWORD *)(a1 + 552);
    v34 = &v57;
  }
  v35 = (__int64 *)(32 * v25 + v33);
  if ( v35 )
  {
    *v35 = *v34;
    v36 = v34[1];
    v34[1] = 0;
    v37 = v58;
    v35[1] = v36;
    v38 = v34[2];
    v34[2] = 0;
    v35[2] = v38;
    v39 = v34[3];
    v34[3] = 0;
    v40 = v60;
    v35[3] = v39;
    v25 = *(unsigned int *)(a1 + 560);
    v41 = v40 - v37;
    result = v25;
    *(_DWORD *)(a1 + 560) = v25 + 1;
    if ( v37 )
    {
      j_j___libc_free_0(v37, v41);
      v25 = (unsigned int)(*(_DWORD *)(a1 + 560) - 1);
      result = v25;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 560) = result + 1;
  }
  *(_DWORD *)(v21 + 8) = result;
LABEL_4:
  v26 = *(_QWORD *)(a1 + 552) + 32 * v25;
  v27 = *(__m128i **)(v26 + 16);
  if ( v27 == *(__m128i **)(v26 + 24) )
    return sub_E782B0((const __m128i **)(v26 + 8), v27, &v61);
  if ( v27 )
  {
    *v27 = _mm_loadu_si128(&v61);
    v27[1] = _mm_loadu_si128(&v62);
    v27[2] = _mm_loadu_si128(&v63);
    v27 = *(__m128i **)(v26 + 16);
  }
  *(_QWORD *)(v26 + 16) = v27 + 3;
  return result;
}
