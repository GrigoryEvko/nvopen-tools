// Function: sub_1D30360
// Address: 0x1d30360
//
__int64 __fastcall sub_1D30360(__int64 a1, __int64 a2, __int64 a3, char a4, int a5, __int64 *a6)
{
  __int64 v8; // rbx
  __int64 result; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r8d
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // edx
  int v24; // r10d
  int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  unsigned int v30; // r15d
  __int64 *v31; // rdi
  __int64 v32; // rcx
  int v33; // r9d
  __int64 *v34; // r8

  if ( a3 )
    *(_BYTE *)(a3 + 26) |= 1u;
  v8 = *(_QWORD *)(a1 + 648);
  if ( a4 )
  {
    result = *(unsigned int *)(v8 + 384);
    if ( (unsigned int)result >= *(_DWORD *)(v8 + 388) )
    {
      sub_16CD150(v8 + 376, (const void *)(v8 + 392), 0, 8, a5, (int)a6);
      result = *(unsigned int *)(v8 + 384);
    }
    *(_QWORD *)(*(_QWORD *)(v8 + 376) + 8 * result) = a2;
    ++*(_DWORD *)(v8 + 384);
  }
  else
  {
    result = *(unsigned int *)(v8 + 112);
    if ( (unsigned int)result >= *(_DWORD *)(v8 + 116) )
    {
      sub_16CD150(v8 + 104, (const void *)(v8 + 120), 0, 8, a5, (int)a6);
      result = *(unsigned int *)(v8 + 112);
    }
    *(_QWORD *)(*(_QWORD *)(v8 + 104) + 8 * result) = a2;
    ++*(_DWORD *)(v8 + 112);
  }
  if ( a3 )
  {
    v10 = *(_DWORD *)(v8 + 720);
    v11 = v8 + 696;
    if ( v10 )
    {
      v12 = v10 - 1;
      v13 = *(_QWORD *)(v8 + 704);
      v14 = (v10 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v15 = (__int64 *)(v13 + 40LL * v14);
      v16 = *v15;
      if ( a3 == *v15 )
      {
LABEL_10:
        v17 = *((unsigned int *)v15 + 4);
        if ( (unsigned int)v17 >= *((_DWORD *)v15 + 5) )
        {
          sub_16CD150((__int64)(v15 + 1), v15 + 3, 0, 8, v12, (int)a6);
          result = v15[1] + 8LL * *((unsigned int *)v15 + 4);
        }
        else
        {
          result = v15[1] + 8 * v17;
        }
LABEL_12:
        *(_QWORD *)result = a2;
        ++*((_DWORD *)v15 + 4);
        return result;
      }
      v24 = 1;
      a6 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 && !a6 )
          a6 = v15;
        v14 = v12 & (v24 + v14);
        v15 = (__int64 *)(v13 + 40LL * v14);
        v16 = *v15;
        if ( a3 == *v15 )
          goto LABEL_10;
        ++v24;
      }
      v25 = *(_DWORD *)(v8 + 712);
      if ( a6 )
        v15 = a6;
      ++*(_QWORD *)(v8 + 696);
      v23 = v25 + 1;
      if ( 4 * (v25 + 1) < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(v8 + 716) - v23 > v10 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(v8 + 712) = v23;
          if ( *v15 != -8 )
            --*(_DWORD *)(v8 + 716);
          result = (__int64)(v15 + 3);
          *v15 = a3;
          v15[1] = (__int64)(v15 + 3);
          v15[2] = 0x200000000LL;
          goto LABEL_12;
        }
        sub_1D30070(v11, v10);
        v26 = *(_DWORD *)(v8 + 720);
        if ( v26 )
        {
          v27 = v26 - 1;
          v28 = *(_QWORD *)(v8 + 704);
          v29 = 1;
          v30 = v27 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v15 = (__int64 *)(v28 + 40LL * v30);
          v23 = *(_DWORD *)(v8 + 712) + 1;
          v31 = 0;
          v32 = *v15;
          if ( a3 != *v15 )
          {
            while ( v32 != -8 )
            {
              if ( v32 == -16 && !v31 )
                v31 = v15;
              v30 = v27 & (v29 + v30);
              v15 = (__int64 *)(v28 + 40LL * v30);
              v32 = *v15;
              if ( a3 == *v15 )
                goto LABEL_21;
              ++v29;
            }
            if ( v31 )
              v15 = v31;
          }
          goto LABEL_21;
        }
LABEL_56:
        ++*(_DWORD *)(v8 + 712);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v8 + 696);
    }
    sub_1D30070(v11, 2 * v10);
    v18 = *(_DWORD *)(v8 + 720);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v8 + 704);
      v21 = (v18 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v15 = (__int64 *)(v20 + 40LL * v21);
      v22 = *v15;
      v23 = *(_DWORD *)(v8 + 712) + 1;
      if ( a3 != *v15 )
      {
        v33 = 1;
        v34 = 0;
        while ( v22 != -8 )
        {
          if ( !v34 && v22 == -16 )
            v34 = v15;
          v21 = v19 & (v33 + v21);
          v15 = (__int64 *)(v20 + 40LL * v21);
          v22 = *v15;
          if ( a3 == *v15 )
            goto LABEL_21;
          ++v33;
        }
        if ( v34 )
          v15 = v34;
      }
      goto LABEL_21;
    }
    goto LABEL_56;
  }
  return result;
}
