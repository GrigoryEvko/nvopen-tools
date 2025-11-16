// Function: sub_39CF220
// Address: 0x39cf220
//
void __fastcall sub_39CF220(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // r12
  unsigned int v9; // esi
  __int64 v10; // r12
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // r9
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rdi
  int v17; // r11d
  _QWORD *v18; // rdx
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // eax
  __int64 v25; // rdi
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdi
  _QWORD *v31; // r8
  unsigned int v32; // r14d
  int v33; // r9d
  __int64 v34; // rsi

  v5 = (_QWORD *)sub_22077B0(0x48u);
  v6 = (__int64)v5;
  if ( v5 )
  {
    *v5 = a2;
    v7 = v5 + 7;
    *(v7 - 6) = 0;
    *(v7 - 5) = 0;
    *((_DWORD *)v7 - 8) = -1;
    *(v7 - 3) = 0;
    *(_QWORD *)(v6 + 40) = v7;
    *(_QWORD *)(v6 + 48) = 0x100000000LL;
  }
  sub_39A0D10(*(_QWORD *)(a1 + 208), a3, v6);
  if ( !sub_39C7370(a1) || (unsigned __int8)sub_3989C80(*(_QWORD *)(a1 + 200)) )
  {
    v8 = *(_QWORD *)(a1 + 208);
    v9 = *(_DWORD *)(v8 + 352);
    v10 = v8 + 328;
    if ( !v9 )
    {
LABEL_22:
      ++*(_QWORD *)v10;
      goto LABEL_23;
    }
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 920);
    v10 = a1 + 896;
    if ( !v9 )
      goto LABEL_22;
  }
  v11 = *(_QWORD *)(v10 + 8);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( *v13 != a2 )
  {
    v17 = 1;
    v18 = 0;
    while ( v14 != -8 )
    {
      if ( v14 == -16 && !v18 )
        v18 = v13;
      v12 = (v9 - 1) & (v17 + v12);
      v13 = (_QWORD *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == a2 )
        goto LABEL_6;
      ++v17;
    }
    if ( !v18 )
      v18 = v13;
    v19 = *(_DWORD *)(v10 + 16);
    ++*(_QWORD *)v10;
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v10 + 20) - v20 > v9 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(v10 + 16) = v20;
        if ( *v18 != -8 )
          --*(_DWORD *)(v10 + 20);
        *v18 = a2;
        v18[1] = v6;
        return;
      }
      sub_39CF010(v10, v9);
      v28 = *(_DWORD *)(v10 + 24);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(v10 + 8);
        v31 = 0;
        v32 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v33 = 1;
        v20 = *(_DWORD *)(v10 + 16) + 1;
        v18 = (_QWORD *)(v30 + 16LL * v32);
        v34 = *v18;
        if ( *v18 != a2 )
        {
          while ( v34 != -8 )
          {
            if ( !v31 && v34 == -16 )
              v31 = v18;
            v32 = v29 & (v33 + v32);
            v18 = (_QWORD *)(v30 + 16LL * v32);
            v34 = *v18;
            if ( *v18 == a2 )
              goto LABEL_16;
            ++v33;
          }
          if ( v31 )
            v18 = v31;
        }
        goto LABEL_16;
      }
LABEL_51:
      ++*(_DWORD *)(v10 + 16);
      BUG();
    }
LABEL_23:
    sub_39CF010(v10, 2 * v9);
    v21 = *(_DWORD *)(v10 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v10 + 8);
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = *(_DWORD *)(v10 + 16) + 1;
      v18 = (_QWORD *)(v23 + 16LL * v24);
      v25 = *v18;
      if ( *v18 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -8 )
        {
          if ( !v27 && v25 == -16 )
            v27 = v18;
          v24 = v22 & (v26 + v24);
          v18 = (_QWORD *)(v23 + 16LL * v24);
          v25 = *v18;
          if ( *v18 == a2 )
            goto LABEL_16;
          ++v26;
        }
        if ( v27 )
          v18 = v27;
      }
      goto LABEL_16;
    }
    goto LABEL_51;
  }
LABEL_6:
  v15 = v13[1];
  v13[1] = v6;
  if ( v15 )
  {
    v16 = *(_QWORD *)(v15 + 40);
    if ( v16 != v15 + 56 )
      _libc_free(v16);
    j_j___libc_free_0(v15);
  }
}
