// Function: sub_27D4B00
// Address: 0x27d4b00
//
__int64 __fastcall sub_27D4B00(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 result; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 i; // rdx
  __int64 *v13; // r13
  __int64 *v14; // r12
  __int64 v15; // r8
  __int64 *v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // esi
  int v19; // eax
  int v20; // ecx
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 *v23; // r10
  __int64 v24; // rdi
  int v25; // edx
  int v26; // r11d
  __int64 *v27; // r9
  int v28; // r11d
  int v29; // eax
  int v30; // eax
  int v31; // ecx
  __int64 v32; // r8
  int v33; // r11d
  unsigned int v34; // eax
  __int64 v35; // rdi

  result = (char *)*a3 - (char *)*a2;
  if ( result <= 0 )
  {
    *(_QWORD *)a1 = 0;
    goto LABEL_24;
  }
  v6 = (result >> 3) - 1;
  if ( v6 )
  {
    _BitScanReverse64(&v6, v6);
    *(_QWORD *)a1 = 0;
    result = 1LL << (64 - ((unsigned __int8)v6 ^ 0x3Fu));
    if ( (_DWORD)result )
    {
      v7 = (((4 * (int)result / 3u + 1) | ((unsigned __int64)(4 * (int)result / 3u + 1) >> 1)) >> 2)
         | (4 * (int)result / 3u + 1)
         | ((unsigned __int64)(4 * (int)result / 3u + 1) >> 1)
         | (((((4 * (int)result / 3u + 1) | ((unsigned __int64)(4 * (int)result / 3u + 1) >> 1)) >> 2)
           | (4 * (int)result / 3u + 1)
           | ((unsigned __int64)(4 * (int)result / 3u + 1) >> 1)) >> 4);
      v8 = ((v7 >> 8) | v7 | (((v7 >> 8) | v7) >> 16)) + 1;
      v9 = ((v7 >> 8) | v7 | (((v7 >> 8) | v7) >> 16)) + 1;
      v10 = 8 * v8;
      goto LABEL_5;
    }
LABEL_24:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_9;
  }
  *(_QWORD *)a1 = 0;
  v9 = 4;
  v10 = 32;
LABEL_5:
  *(_DWORD *)(a1 + 24) = v9;
  result = sub_C7D670(v10, 8);
  v11 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = result;
  for ( i = result + 8 * v11; i != result; result += 8 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
LABEL_9:
  v13 = *a3;
  v14 = *a2;
  if ( v13 != *a2 )
  {
    while ( 1 )
    {
      v18 = *(_DWORD *)(a1 + 24);
      if ( !v18 )
        break;
      v15 = *(_QWORD *)(a1 + 8);
      result = (v18 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v16 = (__int64 *)(v15 + 8 * result);
      v17 = *v16;
      if ( *v14 != *v16 )
      {
        v28 = 1;
        v23 = 0;
        while ( v17 != -4096 )
        {
          if ( v23 || v17 != -8192 )
            v16 = v23;
          result = (v18 - 1) & (v28 + (_DWORD)result);
          v17 = *(_QWORD *)(v15 + 8LL * (unsigned int)result);
          if ( *v14 == v17 )
            goto LABEL_12;
          ++v28;
          v23 = v16;
          v16 = (__int64 *)(v15 + 8LL * (unsigned int)result);
        }
        v29 = *(_DWORD *)(a1 + 16);
        if ( !v23 )
          v23 = v16;
        ++*(_QWORD *)a1;
        v25 = v29 + 1;
        if ( 4 * (v29 + 1) < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a1 + 20) - v25 <= v18 >> 3 )
          {
            sub_27D4930(a1, v18);
            v30 = *(_DWORD *)(a1 + 24);
            if ( !v30 )
            {
LABEL_53:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v31 = v30 - 1;
            v32 = *(_QWORD *)(a1 + 8);
            v27 = 0;
            v33 = 1;
            v34 = (v30 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
            v23 = (__int64 *)(v32 + 8LL * v34);
            v35 = *v23;
            v25 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v23 != *v14 )
            {
              while ( v35 != -4096 )
              {
                if ( v35 == -8192 && !v27 )
                  v27 = v23;
                v34 = v31 & (v33 + v34);
                v23 = (__int64 *)(v32 + 8LL * v34);
                v35 = *v23;
                if ( *v14 == *v23 )
                  goto LABEL_31;
                ++v33;
              }
LABEL_19:
              if ( v27 )
                v23 = v27;
            }
          }
LABEL_31:
          *(_DWORD *)(a1 + 16) = v25;
          if ( *v23 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v14;
          *v23 = *v14;
          goto LABEL_12;
        }
LABEL_15:
        sub_27D4930(a1, 2 * v18);
        v19 = *(_DWORD *)(a1 + 24);
        if ( !v19 )
          goto LABEL_53;
        v20 = v19 - 1;
        v21 = *(_QWORD *)(a1 + 8);
        v22 = (v19 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
        v23 = (__int64 *)(v21 + 8LL * v22);
        v24 = *v23;
        v25 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v23 != *v14 )
        {
          v26 = 1;
          v27 = 0;
          while ( v24 != -4096 )
          {
            if ( v24 == -8192 && !v27 )
              v27 = v23;
            v22 = v20 & (v26 + v22);
            v23 = (__int64 *)(v21 + 8LL * v22);
            v24 = *v23;
            if ( *v14 == *v23 )
              goto LABEL_31;
            ++v26;
          }
          goto LABEL_19;
        }
        goto LABEL_31;
      }
LABEL_12:
      if ( v13 == ++v14 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_15;
  }
  return result;
}
