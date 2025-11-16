// Function: sub_39A0940
// Address: 0x39a0940
//
void __fastcall sub_39A0940(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // r13
  _QWORD *i; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 *v18; // r8
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 *v21; // rcx
  char *v22; // rdx
  int v23; // esi
  void *v24; // rdi
  unsigned int v25; // r9d
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  _DWORD *v29; // r15
  const void *v30; // rsi
  size_t v31; // rdx
  _QWORD *j; // rdx
  __int64 *v33; // [rsp+0h] [rbp-40h]
  __int64 *v34; // [rsp+0h] [rbp-40h]
  unsigned int v35; // [rsp+Ch] [rbp-34h]
  unsigned int v36; // [rsp+Ch] [rbp-34h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = (_QWORD *)sub_22077B0(136LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = v4 + 136 * v3;
    for ( i = &v6[17 * v7]; i != v6; v6 += 17 )
    {
      if ( v6 )
        *v6 = -8;
    }
    v10 = (_QWORD *)(v4 + 72);
    if ( v8 != v4 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 9);
        if ( v11 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 9);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 136LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( !v16 && v19 == -16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 136LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          *v18 = v11;
          v20 = *(v10 - 6);
          v21 = v18 + 2;
          v22 = (char *)(v10 - 7);
          if ( v20 )
          {
            v23 = *((_DWORD *)v10 - 14);
            v18[3] = v20;
            *((_DWORD *)v18 + 4) = v23;
            v18[4] = *(v10 - 5);
            v18[5] = *(v10 - 4);
            *(_QWORD *)(v20 + 8) = v21;
            v18[6] = *(v10 - 3);
            *(v10 - 6) = 0;
            *(v10 - 5) = v22;
            *(v10 - 4) = v22;
            *(v10 - 3) = 0;
          }
          else
          {
            *((_DWORD *)v18 + 4) = 0;
            v18[3] = 0;
            v18[4] = (__int64)v21;
            v18[5] = (__int64)v21;
            v18[6] = 0;
          }
          v24 = v18 + 9;
          v18[7] = (__int64)(v18 + 9);
          v18[8] = 0x800000000LL;
          v25 = *((_DWORD *)v10 - 2);
          if ( v25 && v18 + 7 != v10 - 2 )
          {
            v29 = (_DWORD *)*(v10 - 2);
            if ( v10 == (_QWORD *)v29 )
            {
              v30 = v10;
              v31 = 8LL * v25;
              if ( v25 <= 8 )
                goto LABEL_28;
              v34 = v18;
              v36 = *((_DWORD *)v10 - 2);
              sub_16CD150((__int64)(v18 + 7), v18 + 9, v25, 8, (int)v18, v25);
              v18 = v34;
              v30 = (const void *)*(v10 - 2);
              v25 = v36;
              v31 = 8LL * *((unsigned int *)v10 - 2);
              v24 = (void *)v34[7];
              if ( v31 )
              {
LABEL_28:
                v33 = v18;
                v35 = v25;
                memcpy(v24, v30, v31);
                v18 = v33;
                v25 = v35;
              }
              *((_DWORD *)v18 + 16) = v25;
              *(v29 - 2) = 0;
            }
            else
            {
              v18[7] = (__int64)v29;
              *((_DWORD *)v18 + 16) = *((_DWORD *)v10 - 2);
              *((_DWORD *)v18 + 17) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v26 = *(v10 - 2);
          if ( (_QWORD *)v26 != v10 )
            _libc_free(v26);
          v27 = *(v10 - 6);
          while ( v27 )
          {
            sub_399FE50(*(_QWORD *)(v27 + 24));
            v28 = v27;
            v27 = *(_QWORD *)(v27 + 16);
            j_j___libc_free_0(v28);
          }
        }
        if ( (_QWORD *)v8 == v10 + 8 )
          break;
        v10 += 17;
      }
    }
    j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[17 * *(unsigned int *)(a1 + 24)]; j != v6; v6 += 17 )
    {
      if ( v6 )
        *v6 = -8;
    }
  }
}
