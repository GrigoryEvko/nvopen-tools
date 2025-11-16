// Function: sub_1E898E0
// Address: 0x1e898e0
//
__int64 __fastcall sub_1E898E0(__int64 a1, __int64 a2)
{
  int *v2; // r12
  __int64 result; // rax
  __int64 v4; // r14
  unsigned int v6; // r8d
  unsigned int v7; // edx
  int *v8; // rdi
  int v9; // esi
  __int64 v10; // r13
  int *v11; // r15
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  _DWORD *v14; // rax
  __int64 v15; // rdx
  int *v16; // r8
  _DWORD *k; // rdx
  int *v18; // r13
  _QWORD *m; // rdx
  _QWORD *v20; // rdx
  int *v21; // r9
  int v22; // edx
  int v23; // r10d
  int v24; // eax
  unsigned __int64 v25; // rax
  _DWORD *v26; // rax
  __int64 v27; // rdx
  int *v28; // r8
  _DWORD *i; // rdx
  int *v30; // r13
  __int64 v31; // rdx
  _DWORD *j; // rdx
  __int64 v33; // rdx
  _DWORD *n; // rdx
  int *v35; // [rsp+0h] [rbp-50h]
  int *v36; // [rsp+0h] [rbp-50h]
  _QWORD *v37; // [rsp+8h] [rbp-48h]
  _QWORD *v38; // [rsp+8h] [rbp-48h]
  _QWORD *v39; // [rsp+8h] [rbp-48h]
  _QWORD v40[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = *(int **)a2;
  result = *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)a2 + 4 * result;
  if ( *(_QWORD *)a2 != v4 )
  {
    while ( 1 )
    {
      v10 = *(unsigned int *)(a1 + 24);
      v11 = *(int **)(a1 + 8);
      if ( !(_DWORD)v10 )
        break;
      result = (unsigned int)*v2;
      v6 = v10 - 1;
      v7 = (v10 - 1) & (37 * result);
      v8 = &v11[v7];
      v9 = *v8;
      if ( (_DWORD)result != *v8 )
      {
        v23 = 1;
        v21 = 0;
        while ( v9 != -1 )
        {
          if ( v21 || v9 != -2 )
            v8 = v21;
          v7 = v6 & (v23 + v7);
          v9 = v11[v7];
          if ( (_DWORD)result == v9 )
            goto LABEL_4;
          ++v23;
          v21 = v8;
          v8 = &v11[v7];
        }
        v24 = *(_DWORD *)(a1 + 16);
        if ( !v21 )
          v21 = v8;
        ++*(_QWORD *)a1;
        v22 = v24 + 1;
        if ( 4 * (v24 + 1) < (unsigned int)(3 * v10) )
        {
          if ( (int)v10 - *(_DWORD *)(a1 + 20) - v22 > (unsigned int)v10 >> 3 )
            goto LABEL_24;
          v25 = (((((((((((unsigned __int64)v6 >> 1) | v6) >> 2) | ((unsigned __int64)v6 >> 1) | v6) >> 4)
                   | ((((unsigned __int64)v6 >> 1) | v6) >> 2)
                   | ((unsigned __int64)v6 >> 1)
                   | v6) >> 8)
                 | ((((((unsigned __int64)v6 >> 1) | v6) >> 2) | ((unsigned __int64)v6 >> 1) | v6) >> 4)
                 | ((((unsigned __int64)v6 >> 1) | v6) >> 2)
                 | ((unsigned __int64)v6 >> 1)
                 | v6) >> 16)
               | ((((((((unsigned __int64)v6 >> 1) | v6) >> 2) | ((unsigned __int64)v6 >> 1) | v6) >> 4)
                 | ((((unsigned __int64)v6 >> 1) | v6) >> 2)
                 | ((unsigned __int64)v6 >> 1)
                 | v6) >> 8)
               | ((((((unsigned __int64)v6 >> 1) | v6) >> 2) | ((unsigned __int64)v6 >> 1) | v6) >> 4)
               | ((((unsigned __int64)v6 >> 1) | v6) >> 2)
               | ((unsigned __int64)v6 >> 1)
               | v6)
              + 1;
          if ( (unsigned int)v25 < 0x40 )
            LODWORD(v25) = 64;
          *(_DWORD *)(a1 + 24) = v25;
          v26 = (_DWORD *)sub_22077B0(4LL * (unsigned int)v25);
          *(_QWORD *)(a1 + 8) = v26;
          if ( v11 )
          {
            v27 = *(unsigned int *)(a1 + 24);
            *(_QWORD *)(a1 + 16) = 0;
            v28 = &v11[v10];
            for ( i = &v26[v27]; i != v26; ++v26 )
            {
              if ( v26 )
                *v26 = -1;
            }
            v30 = v11;
            m = v40;
            do
            {
              if ( (unsigned int)*v30 <= 0xFFFFFFFD )
              {
                v36 = v28;
                v39 = m;
                sub_1DF91F0(a1, v30, m);
                v28 = v36;
                m = v39;
                *(_DWORD *)v40[0] = *v30;
                ++*(_DWORD *)(a1 + 16);
              }
              ++v30;
            }
            while ( v28 != v30 );
LABEL_15:
            v37 = m;
            j___libc_free_0(v11);
            v20 = v37;
LABEL_16:
            sub_1DF91F0(a1, v2, v20);
            v21 = (int *)v40[0];
            v22 = *(_DWORD *)(a1 + 16) + 1;
LABEL_24:
            *(_DWORD *)(a1 + 16) = v22;
            if ( *v21 != -1 )
              --*(_DWORD *)(a1 + 20);
            result = (unsigned int)*v2;
            *v21 = result;
            goto LABEL_4;
          }
          v31 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          for ( j = &v26[v31]; j != v26; ++v26 )
          {
            if ( v26 )
              *v26 = -1;
          }
LABEL_49:
          v20 = v40;
          goto LABEL_16;
        }
LABEL_7:
        v12 = ((((((((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v10 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v10 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v10 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v10 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 16;
        v13 = (v12
             | (((((((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v10 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v10 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v10 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v10 - 1) | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v10 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v10 - 1) >> 1))
            + 1;
        if ( (unsigned int)v13 < 0x40 )
          LODWORD(v13) = 64;
        *(_DWORD *)(a1 + 24) = v13;
        v14 = (_DWORD *)sub_22077B0(4LL * (unsigned int)v13);
        *(_QWORD *)(a1 + 8) = v14;
        if ( v11 )
        {
          v15 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          v16 = &v11[v10];
          for ( k = &v14[v15]; k != v14; ++v14 )
          {
            if ( v14 )
              *v14 = -1;
          }
          v18 = v11;
          for ( m = v40; v16 != v18; ++v18 )
          {
            if ( (unsigned int)*v18 <= 0xFFFFFFFD )
            {
              v35 = v16;
              v38 = m;
              sub_1DF91F0(a1, v18, m);
              v16 = v35;
              m = v38;
              *(_DWORD *)v40[0] = *v18;
              ++*(_DWORD *)(a1 + 16);
            }
          }
          goto LABEL_15;
        }
        v33 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        for ( n = &v14[v33]; n != v14; ++v14 )
        {
          if ( v14 )
            *v14 = -1;
        }
        goto LABEL_49;
      }
LABEL_4:
      if ( (int *)v4 == ++v2 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_7;
  }
  return result;
}
