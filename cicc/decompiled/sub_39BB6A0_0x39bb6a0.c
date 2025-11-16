// Function: sub_39BB6A0
// Address: 0x39bb6a0
//
void __fastcall sub_39BB6A0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  _DWORD *v6; // rax
  unsigned __int64 v7; // r15
  _DWORD *i; // rdx
  _DWORD *v9; // rbx
  char *v10; // rax
  unsigned int v11; // eax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  unsigned int *v15; // r9
  int v16; // r10d
  unsigned int v17; // ecx
  unsigned int *v18; // r8
  unsigned int v19; // esi
  void *v20; // rdi
  unsigned int v21; // r9d
  unsigned __int64 v22; // rdi
  _DWORD *v23; // rax
  const void *v24; // rsi
  size_t v25; // rdx
  _DWORD *j; // rdx
  unsigned int *v27; // [rsp+0h] [rbp-40h]
  unsigned int *v28; // [rsp+0h] [rbp-40h]
  unsigned int v29; // [rsp+Ch] [rbp-34h]
  unsigned int v30; // [rsp+Ch] [rbp-34h]

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
  v6 = (_DWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = v4 + 40 * v3;
    for ( i = &v6[10 * *(unsigned int *)(a1 + 24)]; i != v6; v6 += 10 )
    {
      if ( v6 )
        *v6 = -1;
    }
    v9 = (_DWORD *)(v4 + 24);
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v11 = *(v9 - 6);
        if ( v11 > 0xFFFFFFFD )
          goto LABEL_10;
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 8);
        v15 = 0;
        v16 = 1;
        v17 = v13 & (37 * v11);
        v18 = (unsigned int *)(v14 + 40LL * v17);
        v19 = *v18;
        if ( v11 != *v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v15 && v19 == -2 )
              v15 = v18;
            v17 = v13 & (v16 + v17);
            v18 = (unsigned int *)(v14 + 40LL * v17);
            v19 = *v18;
            if ( v11 == *v18 )
              goto LABEL_15;
            ++v16;
          }
          if ( v15 )
            v18 = v15;
        }
LABEL_15:
        *v18 = v11;
        v20 = v18 + 6;
        *((_QWORD *)v18 + 1) = v18 + 6;
        *((_QWORD *)v18 + 2) = 0x200000000LL;
        v21 = *(v9 - 2);
        if ( v18 + 2 != v9 - 4 && v21 )
        {
          v23 = (_DWORD *)*((_QWORD *)v9 - 2);
          if ( v23 == v9 )
          {
            v24 = v9;
            v25 = 8LL * v21;
            if ( v21 <= 2 )
              goto LABEL_23;
            v28 = v18;
            v30 = *(v9 - 2);
            sub_16CD150((__int64)(v18 + 2), v18 + 6, v21, 8, (int)v18, v21);
            v18 = v28;
            v24 = (const void *)*((_QWORD *)v9 - 2);
            v21 = v30;
            v25 = 8LL * (unsigned int)*(v9 - 2);
            v20 = (void *)*((_QWORD *)v28 + 1);
            if ( v25 )
            {
LABEL_23:
              v27 = v18;
              v29 = v21;
              memcpy(v20, v24, v25);
              v18 = v27;
              v21 = v29;
            }
            v18[4] = v21;
            *(v9 - 2) = 0;
          }
          else
          {
            *((_QWORD *)v18 + 1) = v23;
            v18[4] = *(v9 - 2);
            v18[5] = *(v9 - 1);
            *((_QWORD *)v9 - 2) = v9;
            *(v9 - 1) = 0;
            *(v9 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v22 = *((_QWORD *)v9 - 2);
        if ( (_DWORD *)v22 == v9 )
        {
LABEL_10:
          v10 = (char *)(v9 + 10);
          if ( (_DWORD *)v7 == v9 + 4 )
            break;
        }
        else
        {
          _libc_free(v22);
          v10 = (char *)(v9 + 10);
          if ( (_DWORD *)v7 == v9 + 4 )
            break;
        }
        v9 = v10;
      }
    }
    j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[10 * *(unsigned int *)(a1 + 24)]; j != v6; v6 += 10 )
    {
      if ( v6 )
        *v6 = -1;
    }
  }
}
