// Function: sub_F302F0
// Address: 0xf302f0
//
char __fastcall sub_F302F0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rbx
  __int64 v9; // r8
  int v10; // r10d
  unsigned int v11; // r15d
  unsigned int v12; // edi
  __int64 *v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 *v16; // rbx
  __int64 v17; // rax
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // edx
  int v22; // ecx
  __int64 v23; // rdi
  int v24; // ecx
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  __int64 *v28; // r8
  unsigned int v29; // r15d
  int v30; // r9d
  __int64 v31; // rsi
  int v32; // r10d
  __int64 *v33; // r9

  v6 = sub_B326A0((__int64)a2);
  if ( !v6 )
    return v6;
  v7 = *(_DWORD *)(a3 + 24);
  v8 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_9;
  }
  v9 = *(_QWORD *)(a3 + 8);
  v10 = 1;
  v11 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
  v12 = (v7 - 1) & v11;
  v13 = (__int64 *)(v9 + 24LL * v12);
  v14 = 0;
  v15 = *v13;
  if ( v8 != *v13 )
  {
    while ( v15 != -4096 )
    {
      if ( !v14 && v15 == -8192 )
        v14 = v13;
      v12 = (v7 - 1) & (v10 + v12);
      v13 = (__int64 *)(v9 + 24LL * v12);
      v15 = *v13;
      if ( v8 == *v13 )
        goto LABEL_4;
      ++v10;
    }
    v24 = *(_DWORD *)(a3 + 16);
    if ( !v14 )
      v14 = v13;
    ++*(_QWORD *)a3;
    v22 = v24 + 1;
    if ( 4 * v22 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a3 + 20) - v22 > v7 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a3 + 16) = v22;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v14 = v8;
        v16 = v14 + 1;
        v14[1] = 0;
        *((_BYTE *)v14 + 16) = 0;
        v17 = 1;
        goto LABEL_5;
      }
      sub_F300F0(a3, v7);
      v25 = *(_DWORD *)(a3 + 24);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a3 + 8);
        v28 = 0;
        v29 = (v25 - 1) & v11;
        v30 = 1;
        v22 = *(_DWORD *)(a3 + 16) + 1;
        v14 = (__int64 *)(v27 + 24LL * v29);
        v31 = *v14;
        if ( v8 != *v14 )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v28 )
              v28 = v14;
            v29 = v26 & (v30 + v29);
            v14 = (__int64 *)(v27 + 24LL * v29);
            v31 = *v14;
            if ( v8 == *v14 )
              goto LABEL_11;
            ++v30;
          }
          if ( v28 )
            v14 = v28;
        }
        goto LABEL_11;
      }
LABEL_45:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_9:
    sub_F300F0(a3, 2 * v7);
    v18 = *(_DWORD *)(a3 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a3 + 8);
      v21 = (v18 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v22 = *(_DWORD *)(a3 + 16) + 1;
      v14 = (__int64 *)(v20 + 24LL * v21);
      v23 = *v14;
      if ( v8 != *v14 )
      {
        v32 = 1;
        v33 = 0;
        while ( v23 != -4096 )
        {
          if ( !v33 && v23 == -8192 )
            v33 = v14;
          v21 = v19 & (v32 + v21);
          v14 = (__int64 *)(v20 + 24LL * v21);
          v23 = *v14;
          if ( v8 == *v14 )
            goto LABEL_11;
          ++v32;
        }
        if ( v33 )
          v14 = v33;
      }
      goto LABEL_11;
    }
    goto LABEL_45;
  }
LABEL_4:
  v16 = v13 + 1;
  v17 = v13[1] + 1;
LABEL_5:
  *v16 = v17;
  LOBYTE(v6) = sub_F2FDA0(a1, a2);
  if ( (_BYTE)v6 )
    *((_BYTE *)v16 + 8) = 1;
  return v6;
}
