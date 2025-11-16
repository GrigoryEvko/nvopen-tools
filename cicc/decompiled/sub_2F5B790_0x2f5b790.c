// Function: sub_2F5B790
// Address: 0x2f5b790
//
__int64 __fastcall sub_2F5B790(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rdi
  __int64 *v6; // rsi
  __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 *v11; // r12
  __int64 *v12; // r13
  __int64 v13; // r8
  __int64 *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // esi
  int v17; // eax
  int v18; // ecx
  __int64 v19; // r8
  unsigned int v20; // eax
  __int64 *v21; // r10
  __int64 v22; // rdi
  int v23; // edx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r12
  int v27; // r11d
  int v28; // eax
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r8
  __int64 *v32; // r9
  int v33; // r11d
  unsigned int v34; // eax
  __int64 v35; // rdi
  int v36; // r11d
  _BYTE v37[80]; // [rsp+0h] [rbp-50h] BYREF

  if ( *(_DWORD *)(a1 + 16) )
  {
    result = sub_2F5B510((__int64)v37, a1, a2);
    if ( v37[32] )
    {
      result = *(unsigned int *)(a1 + 40);
      v26 = *a2;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v24, v25);
        result = *(unsigned int *)(a1 + 40);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v26;
      ++*(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v4 = *(_QWORD **)(a1 + 32);
    v6 = &v4[*(unsigned int *)(a1 + 40)];
    result = (__int64)sub_2F4C750(v4, (__int64)v6, a2);
    if ( v6 == (__int64 *)result )
    {
      v10 = *a2;
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, v8, v9);
        v6 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
      }
      *v6 = v10;
      result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
      *(_DWORD *)(a1 + 40) = result;
      if ( (unsigned int)result > 8 )
      {
        v11 = *(__int64 **)(a1 + 32);
        v12 = &v11[result];
        while ( 1 )
        {
          v16 = *(_DWORD *)(a1 + 24);
          if ( !v16 )
            break;
          v13 = *(_QWORD *)(a1 + 8);
          result = (v16 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v14 = (__int64 *)(v13 + 8 * result);
          v15 = *v14;
          if ( *v11 != *v14 )
          {
            v27 = 1;
            v21 = 0;
            while ( v15 != -4096 )
            {
              if ( v21 || v15 != -8192 )
                v14 = v21;
              result = (v16 - 1) & (v27 + (_DWORD)result);
              v15 = *(_QWORD *)(v13 + 8LL * (unsigned int)result);
              if ( *v11 == v15 )
                goto LABEL_9;
              ++v27;
              v21 = v14;
              v14 = (__int64 *)(v13 + 8LL * (unsigned int)result);
            }
            v28 = *(_DWORD *)(a1 + 16);
            if ( !v21 )
              v21 = v14;
            ++*(_QWORD *)a1;
            v23 = v28 + 1;
            if ( 4 * (v28 + 1) < 3 * v16 )
            {
              if ( v16 - *(_DWORD *)(a1 + 20) - v23 <= v16 >> 3 )
              {
                sub_2F5B340(a1, v16);
                v29 = *(_DWORD *)(a1 + 24);
                if ( !v29 )
                {
LABEL_50:
                  ++*(_DWORD *)(a1 + 16);
                  BUG();
                }
                v30 = v29 - 1;
                v31 = *(_QWORD *)(a1 + 8);
                v32 = 0;
                v33 = 1;
                v34 = (v29 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
                v21 = (__int64 *)(v31 + 8LL * v34);
                v35 = *v21;
                v23 = *(_DWORD *)(a1 + 16) + 1;
                if ( *v21 != *v11 )
                {
                  while ( v35 != -4096 )
                  {
                    if ( v35 == -8192 && !v32 )
                      v32 = v21;
                    v34 = v30 & (v33 + v34);
                    v21 = (__int64 *)(v31 + 8LL * v34);
                    v35 = *v21;
                    if ( *v11 == *v21 )
                      goto LABEL_14;
                    ++v33;
                  }
LABEL_30:
                  if ( v32 )
                    v21 = v32;
                }
              }
LABEL_14:
              *(_DWORD *)(a1 + 16) = v23;
              if ( *v21 != -4096 )
                --*(_DWORD *)(a1 + 20);
              result = *v11;
              *v21 = *v11;
              goto LABEL_9;
            }
LABEL_12:
            sub_2F5B340(a1, 2 * v16);
            v17 = *(_DWORD *)(a1 + 24);
            if ( !v17 )
              goto LABEL_50;
            v18 = v17 - 1;
            v19 = *(_QWORD *)(a1 + 8);
            v20 = (v17 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
            v21 = (__int64 *)(v19 + 8LL * v20);
            v22 = *v21;
            v23 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v21 != *v11 )
            {
              v36 = 1;
              v32 = 0;
              while ( v22 != -4096 )
              {
                if ( v22 == -8192 && !v32 )
                  v32 = v21;
                v20 = v18 & (v36 + v20);
                v21 = (__int64 *)(v19 + 8LL * v20);
                v22 = *v21;
                if ( *v11 == *v21 )
                  goto LABEL_14;
                ++v36;
              }
              goto LABEL_30;
            }
            goto LABEL_14;
          }
LABEL_9:
          if ( v12 == ++v11 )
            return result;
        }
        ++*(_QWORD *)a1;
        goto LABEL_12;
      }
    }
  }
  return result;
}
