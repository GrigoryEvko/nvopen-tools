// Function: sub_11A2F60
// Address: 0x11a2f60
//
unsigned __int64 __fastcall sub_11A2F60(__int64 a1, __int64 *a2)
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
  _QWORD *v26; // r11
  int v27; // r13d
  unsigned int v28; // edx
  _QWORD *v29; // r8
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
  int v42; // eax
  int v43; // ecx
  unsigned int v44; // edx
  __int64 v45; // rdi
  int v46; // r10d
  int v47; // eax
  int v48; // ecx
  int v49; // r10d
  unsigned int v50; // edx
  __int64 v51; // rdi
  int v52; // r11d

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
      v29 = (_QWORD *)(v25 + 8LL * v28);
      v30 = *v29;
      if ( *a2 == *v29 )
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
        v29 = (_QWORD *)(v25 + 8LL * v28);
      }
      if ( !v26 )
        v26 = v29;
      v31 = result + 1;
      ++*(_QWORD *)a1;
      if ( 4 * v31 < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(a1 + 20) - v31 > v24 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(a1 + 16) = v31;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 20);
          v32 = *a2;
          *v26 = v32;
          result = *(unsigned int *)(a1 + 40);
          if ( result + 1 > *(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, (__int64)v29, v25);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v32;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
        sub_CF4090(a1, v24);
        v47 = *(_DWORD *)(a1 + 24);
        if ( v47 )
        {
          v48 = v47 - 1;
          v29 = *(_QWORD **)(a1 + 8);
          v25 = 0;
          v49 = 1;
          v50 = (v47 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
          v26 = &v29[v50];
          v51 = *v26;
          v31 = *(_DWORD *)(a1 + 16) + 1;
          if ( *a2 == *v26 )
            goto LABEL_24;
          while ( v51 != -4096 )
          {
            if ( !v25 && v51 == -8192 )
              v25 = (__int64)v26;
            v50 = v48 & (v49 + v50);
            v26 = &v29[v50];
            v51 = *v26;
            if ( *a2 == *v26 )
              goto LABEL_24;
            ++v49;
          }
          goto LABEL_46;
        }
        goto LABEL_85;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_CF4090(a1, 2 * v24);
    v42 = *(_DWORD *)(a1 + 24);
    if ( v42 )
    {
      v43 = v42 - 1;
      v29 = *(_QWORD **)(a1 + 8);
      v44 = (v42 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v26 = &v29[v44];
      v45 = *v26;
      v31 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *v26 )
        goto LABEL_24;
      v46 = 1;
      v25 = 0;
      while ( v45 != -4096 )
      {
        if ( v45 == -8192 && !v25 )
          v25 = (__int64)v26;
        v44 = v43 & (v46 + v44);
        v26 = &v29[v44];
        v45 = *v26;
        if ( *a2 == *v26 )
          goto LABEL_24;
        ++v46;
      }
LABEL_46:
      if ( v25 )
        v26 = (_QWORD *)v25;
      goto LABEL_24;
    }
LABEL_85:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  v5 = *(_QWORD **)(a1 + 32);
  v7 = &v5[*(unsigned int *)(a1 + 40)];
  result = (unsigned __int64)sub_11A0650(v5, (__int64)v7, a2);
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
    if ( (unsigned int)result > 0x10 )
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
LABEL_84:
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
            goto LABEL_84;
          v18 = v17 - 1;
          v19 = *(_QWORD *)(a1 + 8);
          v20 = (v17 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v21 = (unsigned __int64 *)(v19 + 8LL * v20);
          v22 = *v21;
          v23 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v11 != *v21 )
          {
            v52 = 1;
            v38 = 0;
            while ( v22 != -4096 )
            {
              if ( v22 == -8192 && !v38 )
                v38 = v21;
              v20 = v18 & (v52 + v20);
              v21 = (unsigned __int64 *)(v19 + 8LL * v20);
              v22 = *v21;
              if ( *v11 == *v21 )
                goto LABEL_14;
              ++v52;
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
