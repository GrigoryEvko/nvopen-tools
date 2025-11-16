// Function: sub_29F9A30
// Address: 0x29f9a30
//
void __fastcall sub_29F9A30(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edi
  unsigned int v7; // ebx
  __int64 v8; // rcx
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 *v11; // r9
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned __int64 *v14; // rax
  int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // eax
  unsigned __int64 *v18; // r10
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // r10
  unsigned int v21; // r14d
  int v22; // r9d
  int v23; // r11d
  int v24; // eax
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // ebx
  unsigned __int64 *v29; // rdi
  int v30; // r11d

  v4 = *(_DWORD *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 8);
    v12 = a1 + 8;
    goto LABEL_16;
  }
  v6 = v4 - 1;
  v7 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v8 = (v4 - 1) & v7;
  v9 = (unsigned __int64 *)(v5 + 8 * v8);
  v10 = *v9;
  v11 = v9;
  if ( *v9 == a2 )
  {
LABEL_3:
    if ( v11 != (unsigned __int64 *)(v5 + 8LL * v4) )
      return;
    v12 = a1 + 8;
  }
  else
  {
    v20 = *v9;
    v21 = v8;
    v22 = 1;
    while ( v20 != -4096 )
    {
      v30 = v22 + 1;
      v21 = v6 & (v21 + v22);
      v11 = (unsigned __int64 *)(v5 + 8LL * v21);
      v20 = *v11;
      if ( *v11 == a2 )
        goto LABEL_3;
      v22 = v30;
    }
    v12 = a1 + 8;
    v7 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
    v8 = v7 & v6;
    v9 = (unsigned __int64 *)(v5 + 8 * v8);
    v10 = *v9;
  }
  if ( v10 != a2 )
  {
    v23 = 1;
    v18 = 0;
    while ( v10 != -4096 )
    {
      if ( v10 != -8192 || v18 )
        v9 = v18;
      v8 = v6 & (v23 + (_DWORD)v8);
      v10 = *(_QWORD *)(v5 + 8LL * (unsigned int)v8);
      if ( v10 == a2 )
        goto LABEL_6;
      ++v23;
      v18 = v9;
      v9 = (unsigned __int64 *)(v5 + 8LL * (unsigned int)v8);
    }
    if ( !v18 )
      v18 = v9;
    v24 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v10 = (unsigned int)(v24 + 1);
    if ( 4 * (int)v10 < 3 * v4 )
    {
      v8 = v4 >> 3;
      if ( v4 - *(_DWORD *)(a1 + 28) - (unsigned int)v10 > (unsigned int)v8 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 24) = v10;
        if ( *v18 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v18 = a2;
        goto LABEL_6;
      }
      sub_22EE7D0(v12, v4);
      v25 = *(_DWORD *)(a1 + 32);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 16);
        v5 = 1;
        v28 = v26 & v7;
        v18 = (unsigned __int64 *)(v27 + 8LL * v28);
        v10 = (unsigned int)(*(_DWORD *)(a1 + 24) + 1);
        v29 = 0;
        v8 = *v18;
        if ( *v18 != a2 )
        {
          while ( v8 != -4096 )
          {
            if ( !v29 && v8 == -8192 )
              v29 = v18;
            v12 = (unsigned int)(v5 + 1);
            v28 = v26 & (v5 + v28);
            v18 = (unsigned __int64 *)(v27 + 8LL * v28);
            v8 = *v18;
            if ( *v18 == a2 )
              goto LABEL_18;
            v5 = (unsigned int)v12;
          }
          if ( v29 )
            v18 = v29;
        }
        goto LABEL_18;
      }
LABEL_58:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_16:
    sub_22EE7D0(v12, 2 * v4);
    v15 = *(_DWORD *)(a1 + 32);
    if ( v15 )
    {
      v8 = (unsigned int)(v15 - 1);
      v16 = *(_QWORD *)(a1 + 16);
      v17 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (unsigned __int64 *)(v16 + 8LL * v17);
      v10 = (unsigned int)(*(_DWORD *)(a1 + 24) + 1);
      v19 = *v18;
      if ( *v18 != a2 )
      {
        v12 = 1;
        v5 = 0;
        while ( v19 != -4096 )
        {
          if ( !v5 && v19 == -8192 )
            v5 = (__int64)v18;
          v17 = v8 & (v12 + v17);
          v18 = (unsigned __int64 *)(v16 + 8LL * v17);
          v19 = *v18;
          if ( *v18 == a2 )
            goto LABEL_18;
          v12 = (unsigned int)(v12 + 1);
        }
        if ( v5 )
          v18 = (unsigned __int64 *)v5;
      }
      goto LABEL_18;
    }
    goto LABEL_58;
  }
LABEL_6:
  if ( *(_QWORD *)(a2 + 16) )
  {
    v13 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v14 = *(unsigned __int64 **)(a2 - 8);
      v8 = (__int64)&v14[v13];
    }
    else
    {
      v8 = a2;
      v14 = (unsigned __int64 *)(a2 - v13 * 8);
    }
    while ( (unsigned __int64 *)v8 != v14 )
    {
      v10 = *v14;
      if ( *(_BYTE *)*v14 > 0x1Cu )
        goto LABEL_13;
      v14 += 4;
    }
    sub_29F8DF0((__int64 *)a1, a2);
  }
  else
  {
LABEL_13:
    sub_29F9D60(a1, a2, v10, v8, v5, v12);
  }
}
