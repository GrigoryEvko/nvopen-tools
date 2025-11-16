// Function: sub_3414EA0
// Address: 0x3414ea0
//
__int64 __fastcall sub_3414EA0(__int64 *a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  __int64 *v12; // r8
  unsigned int v13; // r14d
  unsigned int v14; // edi
  __int64 *v15; // rdx
  __int64 result; // rax
  __int64 v17; // rbx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  int v20; // eax
  __int64 *v21; // rdx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // edx
  unsigned int v28; // r14d
  __int64 *v29; // rbx
  __int64 *i; // r13
  __int64 v31; // rdx
  int v32; // eax
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  __int64 *v36; // r9
  unsigned int v37; // r14d
  int v38; // r10d
  __int64 v39; // rsi
  int v40; // r11d
  __int64 *v41; // r10
  int v42; // [rsp+Ch] [rbp-34h]
  int v43; // [rsp+Ch] [rbp-34h]

  if ( a4 )
  {
    v7 = a1[1];
    v9 = *(_DWORD *)(v7 + 24);
    if ( v9 )
    {
      v10 = *(_QWORD *)(v7 + 8);
      v11 = 1;
      v12 = 0;
      v13 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
      v14 = (v9 - 1) & v13;
      v15 = (__int64 *)(v10 + 8LL * v14);
      result = *v15;
      if ( *v15 == a3 )
        return result;
      while ( result != -4096 )
      {
        if ( result != -8192 || v12 )
          v15 = v12;
        v14 = (v9 - 1) & (v11 + v14);
        result = *(_QWORD *)(v10 + 8LL * v14);
        if ( result == a3 )
          return result;
        ++v11;
        v12 = v15;
        v15 = (__int64 *)(v10 + 8LL * v14);
      }
      v32 = *(_DWORD *)(v7 + 16);
      if ( !v12 )
        v12 = v15;
      ++*(_QWORD *)v7;
      v27 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(v7 + 20) - v27 > v9 >> 3 )
        {
LABEL_12:
          *(_DWORD *)(v7 + 16) = v27;
          if ( *v12 != -4096 )
            --*(_DWORD *)(v7 + 20);
          *v12 = a3;
          v28 = a4 - 1;
          v29 = *(__int64 **)(a3 + 40);
          result = 5LL * *(unsigned int *)(a3 + 64);
          for ( i = &v29[5 * *(unsigned int *)(a3 + 64)]; i != v29; result = sub_3414EA0(a2, a2, v31, v28) )
          {
            v31 = *v29;
            v29 += 5;
          }
          return result;
        }
        v43 = a4;
        sub_3414CD0(v7, v9);
        v33 = *(_DWORD *)(v7 + 24);
        if ( v33 )
        {
          v34 = v33 - 1;
          v35 = *(_QWORD *)(v7 + 8);
          v36 = 0;
          a4 = v43;
          v37 = v34 & v13;
          v38 = 1;
          v12 = (__int64 *)(v35 + 8LL * v37);
          v39 = *v12;
          v27 = *(_DWORD *)(v7 + 16) + 1;
          if ( *v12 != a3 )
          {
            while ( v39 != -4096 )
            {
              if ( !v36 && v39 == -8192 )
                v36 = v12;
              v37 = v34 & (v38 + v37);
              v12 = (__int64 *)(v35 + 8LL * v37);
              v39 = *v12;
              if ( *v12 == a3 )
                goto LABEL_12;
              ++v38;
            }
            if ( v36 )
              v12 = v36;
          }
          goto LABEL_12;
        }
LABEL_51:
        ++*(_DWORD *)(v7 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v7;
    }
    v42 = a4;
    sub_3414CD0(v7, 2 * v9);
    v22 = *(_DWORD *)(v7 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v7 + 8);
      a4 = v42;
      v25 = (v22 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v12 = (__int64 *)(v24 + 8LL * v25);
      v26 = *v12;
      v27 = *(_DWORD *)(v7 + 16) + 1;
      if ( *v12 != a3 )
      {
        v40 = 1;
        v41 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v41 )
            v41 = v12;
          v25 = v23 & (v40 + v25);
          v12 = (__int64 *)(v24 + 8LL * v25);
          v26 = *v12;
          if ( *v12 == a3 )
            goto LABEL_12;
          ++v40;
        }
        if ( v41 )
          v12 = v41;
      }
      goto LABEL_12;
    }
    goto LABEL_51;
  }
  v17 = *a1;
  v18 = *(unsigned int *)(*a1 + 8);
  v19 = *(unsigned int *)(*a1 + 12);
  v20 = *(_DWORD *)(*a1 + 8);
  if ( v18 >= v19 )
  {
    if ( v19 < v18 + 1 )
    {
      sub_C8D5F0(*a1, (const void *)(v17 + 16), v18 + 1, 8u, v18 + 1, a6);
      v18 = *(unsigned int *)(v17 + 8);
    }
    result = *(_QWORD *)v17;
    *(_QWORD *)(*(_QWORD *)v17 + 8 * v18) = a3;
    ++*(_DWORD *)(v17 + 8);
  }
  else
  {
    v21 = (__int64 *)(*(_QWORD *)v17 + 8 * v18);
    if ( v21 )
    {
      *v21 = a3;
      v20 = *(_DWORD *)(v17 + 8);
    }
    result = (unsigned int)(v20 + 1);
    *(_DWORD *)(v17 + 8) = result;
  }
  return result;
}
