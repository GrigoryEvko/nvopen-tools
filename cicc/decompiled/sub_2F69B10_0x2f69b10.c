// Function: sub_2F69B10
// Address: 0x2f69b10
//
__int64 __fastcall sub_2F69B10(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // r11
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r13
  int v7; // r12d
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // r10
  __int64 v11; // r8
  unsigned int v12; // ecx
  __int64 v13; // rax
  int v14; // edx
  const __m128i *v15; // rsi
  unsigned __int64 *v16; // rdi
  __m128i *v17; // rax
  int *v18; // rdi
  int v19; // eax
  int v20; // eax
  int v21; // edx
  int v22; // edx
  __int64 v23; // r8
  unsigned int v24; // ecx
  int v25; // esi
  int v26; // r10d
  int *v27; // r9
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // r8
  int v31; // r10d
  unsigned int v32; // edx
  int v33; // esi
  int v34; // [rsp+10h] [rbp-70h]
  __int64 *v35; // [rsp+10h] [rbp-70h]
  int v36; // [rsp+20h] [rbp-60h]
  __int64 *v37; // [rsp+20h] [rbp-60h]
  __int64 *v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  __m128i v42; // [rsp+40h] [rbp-40h] BYREF

  result = a1[1];
  v39 = *(_QWORD *)result + 8LL * *(unsigned int *)(result + 8);
  if ( *(_QWORD *)result != v39 )
  {
    v3 = *(__int64 **)result;
    while ( 1 )
    {
      v4 = *v3;
      v5 = *(_QWORD *)(*v3 + 32);
      v6 = v5 + 40;
      if ( *(_WORD *)(*v3 + 68) != 14 )
      {
        v6 = v5 + 40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF);
        v5 += 80;
      }
      if ( v6 != v5 )
        break;
LABEL_17:
      if ( (__int64 *)v39 == ++v3 )
      {
        result = a1[1];
        goto LABEL_19;
      }
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v5 )
        goto LABEL_7;
      v7 = *(_DWORD *)(v5 + 8);
      if ( v7 >= 0 )
        goto LABEL_7;
      v8 = *a1;
      v9 = *(_DWORD *)(*a1 + 472LL);
      v10 = *a1 + 448LL;
      if ( !v9 )
        break;
      v11 = *(_QWORD *)(v8 + 456);
      v36 = 37 * v7;
      v12 = (v9 - 1) & (37 * v7);
      v13 = v11 + 32LL * v12;
      v14 = *(_DWORD *)v13;
      if ( v7 != *(_DWORD *)v13 )
      {
        v34 = 1;
        v18 = 0;
        while ( v14 != -1 )
        {
          if ( v14 == -2 && !v18 )
            v18 = (int *)v13;
          v12 = (v9 - 1) & (v34 + v12);
          v13 = v11 + 32LL * v12;
          v14 = *(_DWORD *)v13;
          if ( v7 == *(_DWORD *)v13 )
            goto LABEL_12;
          ++v34;
        }
        if ( !v18 )
          v18 = (int *)v13;
        v19 = *(_DWORD *)(v8 + 464);
        ++*(_QWORD *)(v8 + 448);
        v20 = v19 + 1;
        if ( 4 * v20 < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(v8 + 468) - v20 <= v9 >> 3 )
          {
            v35 = v3;
            sub_2F698F0(v10, v9);
            v28 = *(_DWORD *)(v8 + 472);
            if ( !v28 )
            {
LABEL_56:
              ++*(_DWORD *)(v8 + 464);
              BUG();
            }
            v29 = v28 - 1;
            v30 = *(_QWORD *)(v8 + 456);
            v27 = 0;
            v3 = v35;
            v31 = 1;
            v32 = v29 & v36;
            v20 = *(_DWORD *)(v8 + 464) + 1;
            v18 = (int *)(v30 + 32LL * (v29 & (unsigned int)v36));
            v33 = *v18;
            if ( v7 != *v18 )
            {
              while ( v33 != -1 )
              {
                if ( v33 == -2 && !v27 )
                  v27 = v18;
                v32 = v29 & (v31 + v32);
                v18 = (int *)(v30 + 32LL * v32);
                v33 = *v18;
                if ( v7 == *v18 )
                  goto LABEL_27;
                ++v31;
              }
              goto LABEL_35;
            }
          }
          goto LABEL_27;
        }
LABEL_31:
        v38 = v3;
        sub_2F698F0(v10, 2 * v9);
        v21 = *(_DWORD *)(v8 + 472);
        if ( !v21 )
          goto LABEL_56;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(v8 + 456);
        v3 = v38;
        v24 = v22 & (37 * v7);
        v20 = *(_DWORD *)(v8 + 464) + 1;
        v18 = (int *)(v23 + 32LL * v24);
        v25 = *v18;
        if ( v7 != *v18 )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -1 )
          {
            if ( !v27 && v25 == -2 )
              v27 = v18;
            v24 = v22 & (v26 + v24);
            v18 = (int *)(v23 + 32LL * v24);
            v25 = *v18;
            if ( v7 == *v18 )
              goto LABEL_27;
            ++v26;
          }
LABEL_35:
          if ( v27 )
            v18 = v27;
        }
LABEL_27:
        *(_DWORD *)(v8 + 464) = v20;
        if ( *v18 != -1 )
          --*(_DWORD *)(v8 + 468);
        *v18 = v7;
        v15 = 0;
        v16 = (unsigned __int64 *)(v18 + 2);
        *v16 = 0;
        v16[1] = 0;
        v16[2] = 0;
        goto LABEL_13;
      }
LABEL_12:
      v15 = *(const __m128i **)(v13 + 24);
      v16 = (unsigned __int64 *)(v13 + 8);
LABEL_13:
      v42.m128i_i64[1] = v4;
      v42.m128i_i64[0] = a2;
      v17 = (__m128i *)v16[1];
      if ( v17 == v15 )
      {
        v37 = v3;
        sub_2F69330(v16, v15, &v42);
        v3 = v37;
LABEL_7:
        v5 += 40;
        if ( v6 == v5 )
          goto LABEL_17;
      }
      else
      {
        if ( v17 )
        {
          *v17 = _mm_loadu_si128(&v42);
          v17 = (__m128i *)v16[1];
        }
        v5 += 40;
        v16[1] = (unsigned __int64)&v17[1];
        if ( v6 == v5 )
          goto LABEL_17;
      }
    }
    ++*(_QWORD *)(v8 + 448);
    goto LABEL_31;
  }
LABEL_19:
  *(_DWORD *)(result + 8) = 0;
  return result;
}
