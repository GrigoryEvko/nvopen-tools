// Function: sub_32EAE90
// Address: 0x32eae90
//
__int64 __fastcall sub_32EAE90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rdi
  __int64 *v5; // rsi
  __int64 result; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r12
  __int64 *v10; // r13
  __int64 v11; // r8
  __int64 *v12; // rdi
  __int64 v13; // rcx
  unsigned int v14; // esi
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 *v19; // r10
  __int64 v20; // rdi
  int v21; // edx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r12
  int v25; // r11d
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // r11d
  __int64 *v30; // r9
  __int64 v31; // r8
  unsigned int v32; // eax
  __int64 v33; // rdi
  int v34; // r11d
  __int64 v35; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v36[96]; // [rsp+10h] [rbp-60h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v35 = a2;
  if ( *(_DWORD *)(v2 + 584) )
  {
    result = sub_32B33F0((__int64)v36, v2 + 568, &v35);
    if ( v36[32] )
    {
      result = *(unsigned int *)(v2 + 608);
      v24 = v35;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(v2 + 612) )
      {
        sub_C8D5F0(v2 + 600, (const void *)(v2 + 616), result + 1, 8u, v22, v23);
        result = *(unsigned int *)(v2 + 608);
      }
      *(_QWORD *)(*(_QWORD *)(v2 + 600) + 8 * result) = v24;
      ++*(_DWORD *)(v2 + 608);
    }
  }
  else
  {
    v3 = *(_QWORD **)(v2 + 600);
    v5 = &v3[*(unsigned int *)(v2 + 608)];
    result = (__int64)sub_325EB50(v3, (__int64)v5, &v35);
    if ( v5 == (__int64 *)result )
    {
      if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 612) )
      {
        sub_C8D5F0(v2 + 600, (const void *)(v2 + 616), v7 + 1, 8u, v7, v8);
        v5 = (__int64 *)(*(_QWORD *)(v2 + 600) + 8LL * *(unsigned int *)(v2 + 608));
      }
      *v5 = a2;
      result = (unsigned int)(*(_DWORD *)(v2 + 608) + 1);
      *(_DWORD *)(v2 + 608) = result;
      if ( (unsigned int)result > 0x20 )
      {
        v9 = *(__int64 **)(v2 + 600);
        v10 = &v9[result];
        while ( 1 )
        {
          v14 = *(_DWORD *)(v2 + 592);
          if ( !v14 )
            break;
          v11 = *(_QWORD *)(v2 + 576);
          result = (v14 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
          v12 = (__int64 *)(v11 + 8 * result);
          v13 = *v12;
          if ( *v9 != *v12 )
          {
            v25 = 1;
            v19 = 0;
            while ( v13 != -4096 )
            {
              if ( v19 || v13 != -8192 )
                v12 = v19;
              result = (v14 - 1) & (v25 + (_DWORD)result);
              v13 = *(_QWORD *)(v11 + 8LL * (unsigned int)result);
              if ( *v9 == v13 )
                goto LABEL_9;
              ++v25;
              v19 = v12;
              v12 = (__int64 *)(v11 + 8LL * (unsigned int)result);
            }
            v26 = *(_DWORD *)(v2 + 584);
            if ( !v19 )
              v19 = v12;
            ++*(_QWORD *)(v2 + 568);
            v21 = v26 + 1;
            if ( 4 * (v26 + 1) < 3 * v14 )
            {
              if ( v14 - *(_DWORD *)(v2 + 588) - v21 <= v14 >> 3 )
              {
                sub_32B3220(v2 + 568, v14);
                v27 = *(_DWORD *)(v2 + 592);
                if ( !v27 )
                {
LABEL_50:
                  ++*(_DWORD *)(v2 + 584);
                  BUG();
                }
                v28 = v27 - 1;
                v29 = 1;
                v30 = 0;
                v31 = *(_QWORD *)(v2 + 576);
                v32 = (v27 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
                v19 = (__int64 *)(v31 + 8LL * v32);
                v33 = *v19;
                v21 = *(_DWORD *)(v2 + 584) + 1;
                if ( *v19 != *v9 )
                {
                  while ( v33 != -4096 )
                  {
                    if ( !v30 && v33 == -8192 )
                      v30 = v19;
                    v32 = v28 & (v29 + v32);
                    v19 = (__int64 *)(v31 + 8LL * v32);
                    v33 = *v19;
                    if ( *v9 == *v19 )
                      goto LABEL_14;
                    ++v29;
                  }
LABEL_30:
                  if ( v30 )
                    v19 = v30;
                }
              }
LABEL_14:
              *(_DWORD *)(v2 + 584) = v21;
              if ( *v19 != -4096 )
                --*(_DWORD *)(v2 + 588);
              result = *v9;
              *v19 = *v9;
              goto LABEL_9;
            }
LABEL_12:
            sub_32B3220(v2 + 568, 2 * v14);
            v15 = *(_DWORD *)(v2 + 592);
            if ( !v15 )
              goto LABEL_50;
            v16 = v15 - 1;
            v17 = *(_QWORD *)(v2 + 576);
            v18 = (v15 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
            v19 = (__int64 *)(v17 + 8LL * v18);
            v20 = *v19;
            v21 = *(_DWORD *)(v2 + 584) + 1;
            if ( *v19 != *v9 )
            {
              v34 = 1;
              v30 = 0;
              while ( v20 != -4096 )
              {
                if ( v20 == -8192 && !v30 )
                  v30 = v19;
                v18 = v16 & (v34 + v18);
                v19 = (__int64 *)(v17 + 8LL * v18);
                v20 = *v19;
                if ( *v9 == *v19 )
                  goto LABEL_14;
                ++v34;
              }
              goto LABEL_30;
            }
            goto LABEL_14;
          }
LABEL_9:
          if ( v10 == ++v9 )
            return result;
        }
        ++*(_QWORD *)(v2 + 568);
        goto LABEL_12;
      }
    }
  }
  return result;
}
