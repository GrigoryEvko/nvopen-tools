// Function: sub_BCFF80
// Address: 0xbcff80
//
__int64 __fastcall sub_BCFF80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 *v6; // r13
  __int64 v7; // r8
  __int64 *v8; // rdi
  __int64 v9; // rcx
  unsigned int v10; // esi
  int v11; // eax
  int v12; // ecx
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 *v15; // r10
  __int64 v16; // rdi
  int v17; // edx
  int v18; // r11d
  int v19; // eax
  int v20; // eax
  int v21; // ecx
  __int64 v22; // r8
  __int64 *v23; // r9
  int v24; // r11d
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // r11d

  v2 = *(unsigned int *)(a1 + 40);
  if ( v2 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, a1 + 48, v2 + 1, 8);
    v2 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v2) = a2;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 4 )
  {
    v5 = *(__int64 **)(a1 + 32);
    v6 = &v5[result];
    while ( 1 )
    {
      v10 = *(_DWORD *)(a1 + 24);
      if ( !v10 )
        break;
      v7 = *(_QWORD *)(a1 + 8);
      result = (v10 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
      v8 = (__int64 *)(v7 + 8 * result);
      v9 = *v8;
      if ( *v5 != *v8 )
      {
        v18 = 1;
        v15 = 0;
        while ( v9 != -4096 )
        {
          if ( v15 || v9 != -8192 )
            v8 = v15;
          result = (v10 - 1) & (v18 + (_DWORD)result);
          v9 = *(_QWORD *)(v7 + 8LL * (unsigned int)result);
          if ( *v5 == v9 )
            goto LABEL_7;
          ++v18;
          v15 = v8;
          v8 = (__int64 *)(v7 + 8LL * (unsigned int)result);
        }
        v19 = *(_DWORD *)(a1 + 16);
        if ( !v15 )
          v15 = v8;
        ++*(_QWORD *)a1;
        v17 = v19 + 1;
        if ( 4 * (v19 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 20) - v17 <= v10 >> 3 )
          {
            sub_BCFDB0(a1, v10);
            v20 = *(_DWORD *)(a1 + 24);
            if ( !v20 )
            {
LABEL_44:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v21 = v20 - 1;
            v22 = *(_QWORD *)(a1 + 8);
            v23 = 0;
            v24 = 1;
            v25 = (v20 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
            v15 = (__int64 *)(v22 + 8LL * v25);
            v26 = *v15;
            v17 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v15 != *v5 )
            {
              while ( v26 != -4096 )
              {
                if ( v26 == -8192 && !v23 )
                  v23 = v15;
                v25 = v21 & (v24 + v25);
                v15 = (__int64 *)(v22 + 8LL * v25);
                v26 = *v15;
                if ( *v5 == *v15 )
                  goto LABEL_12;
                ++v24;
              }
LABEL_24:
              if ( v23 )
                v15 = v23;
            }
          }
LABEL_12:
          *(_DWORD *)(a1 + 16) = v17;
          if ( *v15 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v5;
          *v15 = *v5;
          goto LABEL_7;
        }
LABEL_10:
        sub_BCFDB0(a1, 2 * v10);
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
          goto LABEL_44;
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = (v11 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
        v15 = (__int64 *)(v13 + 8LL * v14);
        v16 = *v15;
        v17 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v15 != *v5 )
        {
          v27 = 1;
          v23 = 0;
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v23 )
              v23 = v15;
            v14 = v12 & (v27 + v14);
            v15 = (__int64 *)(v13 + 8LL * v14);
            v16 = *v15;
            if ( *v5 == *v15 )
              goto LABEL_12;
            ++v27;
          }
          goto LABEL_24;
        }
        goto LABEL_12;
      }
LABEL_7:
      if ( v6 == ++v5 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_10;
  }
  return result;
}
