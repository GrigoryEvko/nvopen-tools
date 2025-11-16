// Function: sub_252E280
// Address: 0x252e280
//
unsigned __int64 __fastcall sub_252E280(__int64 a1, __int64 *a2)
{
  unsigned __int64 result; // rax
  _QWORD *v5; // rdi
  __int64 *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // r13
  __int64 v13; // r8
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rcx
  unsigned int v16; // esi
  int v17; // eax
  int v18; // ecx
  __int64 v19; // r8
  unsigned int v20; // eax
  unsigned __int64 *v21; // r10
  unsigned __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // esi
  __int64 v25; // r9
  __int64 *v26; // r11
  int v27; // r13d
  unsigned int v28; // edx
  __int64 *v29; // r8
  __int64 v30; // rdi
  int v31; // eax
  __int64 v32; // r12
  int v33; // r11d
  int v34; // eax
  int v35; // eax
  int v36; // ecx
  __int64 v37; // r8
  unsigned __int64 *v38; // r9
  int v39; // r11d
  unsigned int v40; // eax
  unsigned __int64 v41; // rdi
  int v42; // r11d
  __int64 *v43; // [rsp+8h] [rbp-28h] BYREF

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = *(_QWORD *)(a1 + 8);
      v26 = 0;
      v27 = 1;
      v28 = (v24 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v29 = (__int64 *)(v25 + 8LL * v28);
      v30 = *v29;
      if ( *v29 == *a2 )
        return result;
      while ( v30 != -4096 )
      {
        if ( v26 || v30 != -8192 )
          v29 = v26;
        v28 = (v24 - 1) & (v27 + v28);
        v30 = *(_QWORD *)(v25 + 8LL * v28);
        if ( *a2 == v30 )
          return result;
        ++v27;
        v26 = v29;
        v29 = (__int64 *)(v25 + 8LL * v28);
      }
      if ( !v26 )
        v26 = v29;
      v31 = result + 1;
      ++*(_QWORD *)a1;
      v43 = v26;
      if ( 4 * v31 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 20) - v31 > v24 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(a1 + 16) = v31;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v26 = *a2;
          result = *(unsigned int *)(a1 + 40);
          v32 = *a2;
          if ( result + 1 > *(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, (__int64)v29, v25);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v32;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
LABEL_43:
        sub_CF4090(a1, v24);
        sub_23FDF60(a1, a2, &v43);
        v26 = v43;
        v31 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_24;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v43 = 0;
    }
    v24 *= 2;
    goto LABEL_43;
  }
  v5 = *(_QWORD **)(a1 + 32);
  v7 = &v5[*(unsigned int *)(a1 + 40)];
  result = (unsigned __int64)sub_2506500(v5, (__int64)v7, a2);
  if ( v7 == (__int64 *)result )
  {
    v10 = *a2;
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, v8, v9);
      v7 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
    }
    *v7 = v10;
    result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( (unsigned int)result > 8 )
    {
      v11 = *(unsigned __int64 **)(a1 + 32);
      v12 = &v11[result];
      while ( 1 )
      {
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          break;
        v13 = *(_QWORD *)(a1 + 8);
        result = (v16 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
        v14 = (unsigned __int64 *)(v13 + 8 * result);
        v15 = *v14;
        if ( *v11 != *v14 )
        {
          v33 = 1;
          v21 = 0;
          while ( v15 != -4096 )
          {
            if ( v21 || v15 != -8192 )
              v14 = v21;
            result = (v16 - 1) & (v33 + (_DWORD)result);
            v15 = *(_QWORD *)(v13 + 8LL * (unsigned int)result);
            if ( *v11 == v15 )
              goto LABEL_9;
            ++v33;
            v21 = v14;
            v14 = (unsigned __int64 *)(v13 + 8LL * (unsigned int)result);
          }
          v34 = *(_DWORD *)(a1 + 16);
          if ( !v21 )
            v21 = v14;
          ++*(_QWORD *)a1;
          v23 = v34 + 1;
          if ( 4 * (v34 + 1) < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(a1 + 20) - v23 <= v16 >> 3 )
            {
              sub_CF4090(a1, v16);
              v35 = *(_DWORD *)(a1 + 24);
              if ( !v35 )
              {
LABEL_66:
                ++*(_DWORD *)(a1 + 16);
                BUG();
              }
              v36 = v35 - 1;
              v37 = *(_QWORD *)(a1 + 8);
              v38 = 0;
              v39 = 1;
              v40 = (v35 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
              v21 = (unsigned __int64 *)(v37 + 8LL * v40);
              v41 = *v21;
              v23 = *(_DWORD *)(a1 + 16) + 1;
              if ( *v11 != *v21 )
              {
                while ( v41 != -4096 )
                {
                  if ( !v38 && v41 == -8192 )
                    v38 = v21;
                  v40 = v36 & (v39 + v40);
                  v21 = (unsigned __int64 *)(v37 + 8LL * v40);
                  v41 = *v21;
                  if ( *v11 == *v21 )
                    goto LABEL_14;
                  ++v39;
                }
LABEL_38:
                if ( v38 )
                  v21 = v38;
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
          sub_CF4090(a1, 2 * v16);
          v17 = *(_DWORD *)(a1 + 24);
          if ( !v17 )
            goto LABEL_66;
          v18 = v17 - 1;
          v19 = *(_QWORD *)(a1 + 8);
          v20 = (v17 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v21 = (unsigned __int64 *)(v19 + 8LL * v20);
          v22 = *v21;
          v23 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v11 != *v21 )
          {
            v42 = 1;
            v38 = 0;
            while ( v22 != -4096 )
            {
              if ( v22 == -8192 && !v38 )
                v38 = v21;
              v20 = v18 & (v42 + v20);
              v21 = (unsigned __int64 *)(v19 + 8LL * v20);
              v22 = *v21;
              if ( *v11 == *v21 )
                goto LABEL_14;
              ++v42;
            }
            goto LABEL_38;
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
  return result;
}
