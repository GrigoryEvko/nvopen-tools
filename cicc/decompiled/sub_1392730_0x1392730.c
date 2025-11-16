// Function: sub_1392730
// Address: 0x1392730
//
__int64 __fastcall sub_1392730(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  int v4; // r15d
  __int32 v5; // r14d
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r11d
  __int64 *v12; // r8
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rcx
  __int64 result; // rax
  unsigned int i; // r9d
  __int64 *v17; // rdi
  __int64 v18; // rcx
  unsigned int v19; // r9d
  __int64 v20; // rcx
  unsigned int *v21; // rdi
  __int64 v22; // rax
  unsigned int *v23; // rsi
  __int64 v24; // rdx
  unsigned int *v25; // rsi
  unsigned int v26; // edx
  unsigned int *v27; // rbx
  unsigned int *v28; // rdx
  unsigned int v29; // edx
  unsigned int *v30; // rdi
  unsigned int v31; // edi
  unsigned int v32; // r14d
  unsigned int v33; // r13d
  int v34; // edi
  int v35; // ecx
  __m128i v36; // xmm0
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // rdx
  int v40; // r9d
  __int64 *v41; // rdi
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rsi
  unsigned int j; // eax
  __int64 v45; // rsi
  unsigned int v46; // eax
  int v47; // edx
  int v48; // edx
  __int64 v49; // rsi
  int v50; // r9d
  unsigned int k; // eax
  __int64 v52; // rcx
  unsigned int v53; // eax
  int v54; // [rsp+8h] [rbp-58h]
  __m128i v55; // [rsp+10h] [rbp-50h] BYREF

  v4 = a3;
  v5 = a3;
  v8 = a4;
  v55.m128i_i64[0] = a2;
  v9 = *(_DWORD *)(a1 + 24);
  v55.m128i_i64[1] = a3;
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_35;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = ((((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v5) << 32)) >> 22)
      ^ (((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v5) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  result = ((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - (v14 << 27));
  for ( i = result & (v9 - 1); ; i = (v9 - 1) & v19 )
  {
    v17 = (__int64 *)(v10 + 24LL * i);
    v18 = *v17;
    if ( *v17 == a2 && *((_DWORD *)v17 + 2) == v4 )
    {
      v20 = *(_QWORD *)(a1 + 32);
      v21 = (unsigned int *)(v20 + 32LL * *((unsigned int *)v17 + 4));
      v22 = v21[6];
      v23 = v21;
      v24 = (unsigned int)v22;
      if ( (_DWORD)v22 != -1 )
      {
        do
        {
          v25 = (unsigned int *)(v20 + 32 * v24);
          v24 = v25[6];
        }
        while ( (_DWORD)v24 != -1 );
        v26 = *v25;
        while ( 1 )
        {
          v21[6] = v26;
          v23 = (unsigned int *)(v20 + 32 * v22);
          v20 = *(_QWORD *)(a1 + 32);
          v22 = v23[6];
          if ( (_DWORD)v22 == -1 )
            break;
          v21 = v23;
        }
      }
      v27 = (unsigned int *)(v20 + 32 * v8);
      result = v27[6];
      v28 = v27;
      if ( (_DWORD)result != -1 )
      {
        v29 = v27[6];
        do
        {
          v30 = (unsigned int *)(v20 + 32LL * v29);
          v29 = v30[6];
        }
        while ( v29 != -1 );
        v31 = *v30;
        while ( 1 )
        {
          v27[6] = v31;
          v28 = (unsigned int *)(v20 + 32LL * (unsigned int)result);
          result = v28[6];
          if ( (_DWORD)result == -1 )
            break;
          v20 = *(_QWORD *)(a1 + 32);
          v27 = v28;
        }
      }
      if ( v28 != v23 )
      {
        v32 = *v28;
        v33 = *v23;
        result = sub_138F7D0(a1, *v23, *v28);
        if ( !(_BYTE)result )
        {
          result = sub_138F7D0(a1, v32, v33);
          if ( !(_BYTE)result )
            return sub_138FA90(a1, v33, v32);
        }
      }
      return result;
    }
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && *((_DWORD *)v17 + 2) == -2 && !v12 )
      v12 = (__int64 *)(v10 + 24LL * i);
LABEL_6:
    v19 = v11 + i;
    ++v11;
  }
  if ( *((_DWORD *)v17 + 2) != -1 )
    goto LABEL_6;
  if ( !v12 )
    v12 = (__int64 *)(v10 + 24LL * i);
  v34 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v35 = v34 + 1;
  if ( 4 * (v34 + 1) >= 3 * v9 )
  {
LABEL_35:
    sub_1392470(a1, 2 * v9);
    v37 = *(_DWORD *)(a1 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v40 = 1;
      v41 = 0;
      v42 = ((((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v5) << 32)) >> 22)
          ^ (((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v5) << 32));
      v43 = ((9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13)))) >> 15)
          ^ (9 * (((v42 - 1 - (v42 << 13)) >> 8) ^ (v42 - 1 - (v42 << 13))));
      for ( j = v38 & (((v43 - 1 - (v43 << 27)) >> 31) ^ (v43 - 1 - ((_DWORD)v43 << 27))); ; j = v38 & v46 )
      {
        v39 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v39 + 24LL * j);
        v45 = *v12;
        if ( *v12 == a2 && *((_DWORD *)v12 + 2) == v4 )
          break;
        if ( v45 == -8 )
        {
          if ( *((_DWORD *)v12 + 2) == -1 )
          {
LABEL_56:
            result = *(unsigned int *)(a1 + 16);
            if ( v41 )
              v12 = v41;
            v35 = result + 1;
            goto LABEL_28;
          }
        }
        else if ( v45 == -16 && *((_DWORD *)v12 + 2) == -2 && !v41 )
        {
          v41 = (__int64 *)(v39 + 24LL * j);
        }
        v46 = v40 + j;
        ++v40;
      }
      goto LABEL_48;
    }
LABEL_69:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v35 <= v9 >> 3 )
  {
    v54 = result;
    sub_1392470(a1, v9);
    v47 = *(_DWORD *)(a1 + 24);
    if ( v47 )
    {
      v48 = v47 - 1;
      v41 = 0;
      v50 = 1;
      for ( k = v48 & v54; ; k = v48 & v53 )
      {
        v49 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v49 + 24LL * k);
        v52 = *v12;
        if ( *v12 == a2 && *((_DWORD *)v12 + 2) == v4 )
          break;
        if ( v52 == -8 )
        {
          if ( *((_DWORD *)v12 + 2) == -1 )
            goto LABEL_56;
        }
        else if ( v52 == -16 && *((_DWORD *)v12 + 2) == -2 && !v41 )
        {
          v41 = (__int64 *)(v49 + 24LL * k);
        }
        v53 = v50 + k;
        ++v50;
      }
LABEL_48:
      result = *(unsigned int *)(a1 + 16);
      v35 = result + 1;
      goto LABEL_28;
    }
    goto LABEL_69;
  }
LABEL_28:
  *(_DWORD *)(a1 + 16) = v35;
  if ( *v12 != -8 || *((_DWORD *)v12 + 2) != -1 )
    --*(_DWORD *)(a1 + 20);
  v55.m128i_i64[0] = a2;
  v55.m128i_i32[2] = v5;
  v36 = _mm_loadu_si128(&v55);
  *((_DWORD *)v12 + 4) = v8;
  *(__m128i *)v12 = v36;
  return result;
}
