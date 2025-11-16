// Function: sub_19D91E0
// Address: 0x19d91e0
//
void __fastcall sub_19D91E0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // r12
  unsigned int v6; // eax
  unsigned __int64 v7; // rdi
  __int64 v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *i; // rdx
  __int64 *v12; // r14
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned int v15; // eax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // esi
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // eax
  _QWORD *v23; // r10
  __int64 v24; // rdi
  int v25; // ecx
  int v26; // r11d
  _QWORD *v27; // r9
  int v28; // r11d
  int v29; // eax
  int v30; // eax
  int v31; // esi
  __int64 v32; // r8
  _QWORD *v33; // r9
  int v34; // r11d
  unsigned int v35; // eax

  v4 = a2;
  *(_QWORD *)a1 = 0;
  if ( (_DWORD)a3 )
  {
    v6 = 4 * a3;
    v7 = (((((((v6 / 3 + 1) | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 2)
           | (v6 / 3 + 1)
           | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 4)
         | (((v6 / 3 + 1) | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 2)
         | (v6 / 3 + 1)
         | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 8)
       | (((((v6 / 3 + 1) | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 2)
         | (v6 / 3 + 1)
         | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 4)
       | (((v6 / 3 + 1) | ((unsigned __int64)(v6 / 3 + 1) >> 1)) >> 2)
       | (v6 / 3 + 1)
       | ((unsigned __int64)(v6 / 3 + 1) >> 1);
    v8 = ((v7 >> 16) | v7) + 1;
    *(_DWORD *)(a1 + 24) = v8;
    v9 = (_QWORD *)sub_22077B0(8 * v8);
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v9;
    for ( i = &v9[v10]; i != v9; ++v9 )
    {
      if ( v9 )
        *v9 = -8;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
  }
  v12 = &a2[a3];
  if ( v12 != a2 )
  {
    while ( 1 )
    {
      v18 = *(_DWORD *)(a1 + 24);
      if ( !v18 )
        break;
      v13 = *v4;
      v14 = *(_QWORD *)(a1 + 8);
      v15 = (v18 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v16 = (_QWORD *)(v14 + 8LL * v15);
      v17 = *v16;
      if ( *v16 != *v4 )
      {
        v28 = 1;
        v23 = 0;
        while ( v17 != -8 )
        {
          if ( v23 || v17 != -16 )
            v16 = v23;
          v15 = (v18 - 1) & (v28 + v15);
          v17 = *(_QWORD *)(v14 + 8LL * v15);
          if ( v13 == v17 )
            goto LABEL_9;
          ++v28;
          v23 = v16;
          v16 = (_QWORD *)(v14 + 8LL * v15);
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
            sub_1467110(a1, v18);
            v30 = *(_DWORD *)(a1 + 24);
            if ( !v30 )
            {
LABEL_51:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v31 = v30 - 1;
            v32 = *(_QWORD *)(a1 + 8);
            v33 = 0;
            v34 = 1;
            v25 = *(_DWORD *)(a1 + 16) + 1;
            v35 = (v30 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
            v23 = (_QWORD *)(v32 + 8LL * v35);
            v13 = *v23;
            if ( *v4 != *v23 )
            {
              while ( v13 != -8 )
              {
                if ( v13 == -16 && !v33 )
                  v33 = v23;
                v35 = v31 & (v34 + v35);
                v23 = (_QWORD *)(v32 + 8LL * v35);
                v13 = *v23;
                if ( *v4 == *v23 )
                  goto LABEL_27;
                ++v34;
              }
              v13 = *v4;
              if ( v33 )
                v23 = v33;
            }
          }
          goto LABEL_27;
        }
LABEL_12:
        sub_1467110(a1, 2 * v18);
        v19 = *(_DWORD *)(a1 + 24);
        if ( !v19 )
          goto LABEL_51;
        v13 = *v4;
        v20 = v19 - 1;
        v21 = *(_QWORD *)(a1 + 8);
        v22 = (v19 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        v23 = (_QWORD *)(v21 + 8LL * v22);
        v24 = *v23;
        v25 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v23 != *v4 )
        {
          v26 = 1;
          v27 = 0;
          while ( v24 != -8 )
          {
            if ( v24 == -16 && !v27 )
              v27 = v23;
            v22 = v20 & (v26 + v22);
            v23 = (_QWORD *)(v21 + 8LL * v22);
            v24 = *v23;
            if ( v13 == *v23 )
              goto LABEL_27;
            ++v26;
          }
          if ( v27 )
            v23 = v27;
        }
LABEL_27:
        *(_DWORD *)(a1 + 16) = v25;
        if ( *v23 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v23 = v13;
      }
LABEL_9:
      if ( v12 == ++v4 )
        return;
    }
    ++*(_QWORD *)a1;
    goto LABEL_12;
  }
}
