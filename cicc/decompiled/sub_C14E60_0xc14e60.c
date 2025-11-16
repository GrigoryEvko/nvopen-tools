// Function: sub_C14E60
// Address: 0xc14e60
//
__int64 __fastcall sub_C14E60(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  int v8; // r11d
  __int64 v9; // rcx
  __int64 v10; // r14
  unsigned int v11; // edx
  __int64 result; // rax
  __int64 v13; // r9
  __int64 v14; // rdi
  __int64 v15; // rdi
  __m128i *v16; // rsi
  int v17; // eax
  int v18; // edx
  unsigned __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 *v22; // rcx
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  int v30; // eax
  int v31; // ecx
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rsi
  int v35; // r9d
  __int64 v36; // r8
  unsigned __int64 v37; // r13
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rdi
  unsigned int v42; // r13d
  int v43; // r8d
  __int64 v44; // rcx
  unsigned __int128 v45; // [rsp+0h] [rbp-50h] BYREF
  __int64 v46; // [rsp+10h] [rbp-40h] BYREF
  __int64 v47; // [rsp+18h] [rbp-38h]
  __int64 v48; // [rsp+20h] [rbp-30h]
  __int64 v49; // [rsp+28h] [rbp-28h]

  v6 = a1 + 328;
  v7 = *(_DWORD *)(a1 + 352);
  v45 = __PAIR128__(a4, a3);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 328);
    goto LABEL_29;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 336);
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v9 + 16LL * v11;
  v13 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
  {
LABEL_3:
    v14 = *(unsigned int *)(result + 8);
    goto LABEL_4;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = result;
    v11 = (v7 - 1) & (v8 + v11);
    result = v9 + 16LL * v11;
    v13 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v10 )
    v10 = result;
  v17 = *(_DWORD *)(a1 + 344);
  ++*(_QWORD *)(a1 + 328);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v7 )
  {
LABEL_29:
    sub_C14C80(v6, 2 * v7);
    v30 = *(_DWORD *)(a1 + 352);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 336);
      v33 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 344) + 1;
      v10 = v32 + 16LL * v33;
      v34 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != a2 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -4096 )
        {
          if ( !v36 && v34 == -8192 )
            v36 = v10;
          v33 = v31 & (v35 + v33);
          v10 = v32 + 16LL * v33;
          v34 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 == a2 )
            goto LABEL_18;
          ++v35;
        }
        if ( v36 )
          v10 = v36;
      }
      goto LABEL_18;
    }
    goto LABEL_56;
  }
  if ( v7 - *(_DWORD *)(a1 + 348) - v18 <= v7 >> 3 )
  {
    sub_C14C80(v6, v7);
    v38 = *(_DWORD *)(a1 + 352);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 336);
      v41 = 0;
      v42 = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v43 = 1;
      v18 = *(_DWORD *)(a1 + 344) + 1;
      v10 = v40 + 16LL * v42;
      v44 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 != a2 )
      {
        while ( v44 != -4096 )
        {
          if ( !v41 && v44 == -8192 )
            v41 = v10;
          v42 = v39 & (v43 + v42);
          v10 = v40 + 16LL * v42;
          v44 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 == a2 )
            goto LABEL_18;
          ++v43;
        }
        if ( v41 )
          v10 = v41;
      }
      goto LABEL_18;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 344);
    BUG();
  }
LABEL_18:
  *(_DWORD *)(a1 + 344) = v18;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 348);
  *(_QWORD *)v10 = a2;
  *(_DWORD *)(v10 + 8) = 0;
  v14 = *(unsigned int *)(a1 + 368);
  v19 = *(unsigned int *)(a1 + 372);
  v46 = a2;
  v20 = v14 + 1;
  v47 = 0;
  result = v14;
  v48 = 0;
  v49 = 0;
  if ( v14 + 1 > v19 )
  {
    v37 = *(_QWORD *)(a1 + 360);
    if ( v37 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v37 + 32 * v14 )
    {
      sub_C14B80(a1 + 360, v20);
      v14 = *(unsigned int *)(a1 + 368);
      v21 = *(_QWORD *)(a1 + 360);
      v22 = &v46;
      result = v14;
    }
    else
    {
      sub_C14B80(a1 + 360, v20);
      v21 = *(_QWORD *)(a1 + 360);
      v14 = *(unsigned int *)(a1 + 368);
      v22 = (__int64 *)((char *)&v46 + v21 - v37);
      result = v14;
    }
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 360);
    v22 = &v46;
  }
  v23 = (__int64 *)(32 * v14 + v21);
  if ( v23 )
  {
    *v23 = *v22;
    v24 = v22[1];
    v22[1] = 0;
    v25 = v47;
    v23[1] = v24;
    v26 = v22[2];
    v22[2] = 0;
    v23[2] = v26;
    v27 = v22[3];
    v22[3] = 0;
    v28 = v49;
    v23[3] = v27;
    v14 = *(unsigned int *)(a1 + 368);
    v29 = v28 - v25;
    result = v14;
    *(_DWORD *)(a1 + 368) = v14 + 1;
    if ( v25 )
    {
      j_j___libc_free_0(v25, v29);
      v14 = (unsigned int)(*(_DWORD *)(a1 + 368) - 1);
      result = v14;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 368) = result + 1;
  }
  *(_DWORD *)(v10 + 8) = result;
LABEL_4:
  v15 = *(_QWORD *)(a1 + 360) + 32 * v14;
  v16 = *(__m128i **)(v15 + 16);
  if ( v16 == *(__m128i **)(v15 + 24) )
    return sub_A04210((const __m128i **)(v15 + 8), v16, (const __m128i *)&v45);
  if ( v16 )
  {
    *v16 = _mm_load_si128((const __m128i *)&v45);
    v16 = *(__m128i **)(v15 + 16);
  }
  *(_QWORD *)(v15 + 16) = v16 + 1;
  return result;
}
