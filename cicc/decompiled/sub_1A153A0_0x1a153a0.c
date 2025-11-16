// Function: sub_1A153A0
// Address: 0x1a153a0
//
__int64 __fastcall sub_1A153A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  __int64 v8; // rdx
  int v9; // r11d
  __int64 *v10; // r10
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned int i; // r8d
  __int64 *v15; // rcx
  __int64 v16; // r14
  unsigned int v17; // r8d
  int v19; // edx
  int v20; // r8d
  __int64 v21; // r9
  __int64 *v22; // rax
  char v23; // dl
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 v28; // rbx
  __int64 *v29; // rsi
  unsigned int v30; // edi
  __int64 *v31; // rcx
  __int64 v32; // rax
  int v33; // ecx
  __int64 v34; // rdx
  int v35; // edi
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rsi
  int v38; // eax
  __int64 *v39; // rsi
  unsigned int j; // eax
  __int64 v41; // r8
  unsigned int v42; // eax
  int v43; // edx
  int v44; // edx
  int v45; // edi
  unsigned int k; // eax
  __int64 v47; // r8
  unsigned int v48; // eax
  int v49; // [rsp+8h] [rbp-28h]

  v5 = a1 + 2400;
  v7 = *(_DWORD *)(a1 + 2424);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 2400);
    goto LABEL_47;
  }
  v9 = 1;
  v10 = 0;
  v11 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  v13 = ((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - (v12 << 27));
  for ( i = v13 & (v7 - 1); ; i = (v7 - 1) & v17 )
  {
    v8 = *(_QWORD *)(a1 + 2408);
    v15 = (__int64 *)(v8 + 16LL * i);
    v16 = *v15;
    if ( a2 == *v15 && a3 == v15[1] )
      return 0;
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && v15[1] == -16 && !v10 )
      v10 = (__int64 *)(v8 + 16LL * i);
LABEL_9:
    v17 = v9 + i;
    ++v9;
  }
  if ( v15[1] != -8 )
    goto LABEL_9;
  v19 = *(_DWORD *)(a1 + 2416);
  if ( v10 )
    v15 = v10;
  ++*(_QWORD *)(a1 + 2400);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v7 )
  {
LABEL_47:
    sub_1A15100(v5, 2 * v7);
    v33 = *(_DWORD *)(a1 + 2424);
    if ( v33 )
    {
      v34 = *(_QWORD *)(a1 + 2408);
      LODWORD(v21) = v33 - 1;
      v35 = 1;
      v36 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
             | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
            | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
      v37 = ((9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13)))) >> 15)
          ^ (9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13))));
      v38 = ((v37 - 1 - (v37 << 27)) >> 31) ^ (v37 - 1 - ((_DWORD)v37 << 27));
      v39 = 0;
      for ( j = (v33 - 1) & v38; ; j = v21 & v42 )
      {
        v15 = (__int64 *)(v34 + 16LL * j);
        v41 = *v15;
        if ( a2 == *v15 && a3 == v15[1] )
          break;
        if ( v41 == -8 )
        {
          if ( v15[1] == -8 )
          {
LABEL_70:
            if ( v39 )
              v15 = v39;
            v20 = *(_DWORD *)(a1 + 2416) + 1;
            goto LABEL_17;
          }
        }
        else if ( v41 == -16 && v15[1] == -16 && !v39 )
        {
          v39 = (__int64 *)(v34 + 16LL * j);
        }
        v42 = v35 + j;
        ++v35;
      }
      goto LABEL_66;
    }
LABEL_75:
    ++*(_DWORD *)(a1 + 2416);
    BUG();
  }
  LODWORD(v21) = v7 >> 3;
  if ( v7 - *(_DWORD *)(a1 + 2420) - v20 <= v7 >> 3 )
  {
    v49 = v13;
    sub_1A15100(v5, v7);
    v43 = *(_DWORD *)(a1 + 2424);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = 1;
      v39 = 0;
      v21 = *(_QWORD *)(a1 + 2408);
      for ( k = v44 & v49; ; k = v44 & v48 )
      {
        v15 = (__int64 *)(v21 + 16LL * k);
        v47 = *v15;
        if ( a2 == *v15 && a3 == v15[1] )
          break;
        if ( v47 == -8 )
        {
          if ( v15[1] == -8 )
            goto LABEL_70;
        }
        else if ( v47 == -16 && v15[1] == -16 && !v39 )
        {
          v39 = (__int64 *)(v21 + 16LL * k);
        }
        v48 = v45 + k;
        ++v45;
      }
LABEL_66:
      v20 = *(_DWORD *)(a1 + 2416) + 1;
      goto LABEL_17;
    }
    goto LABEL_75;
  }
LABEL_17:
  *(_DWORD *)(a1 + 2416) = v20;
  if ( *v15 != -8 || v15[1] != -8 )
    --*(_DWORD *)(a1 + 2420);
  *v15 = a2;
  v15[1] = a3;
  v22 = *(__int64 **)(a1 + 24);
  if ( *(__int64 **)(a1 + 32) == v22 )
  {
    v29 = &v22[*(unsigned int *)(a1 + 44)];
    v30 = *(_DWORD *)(a1 + 44);
    if ( v22 == v29 )
    {
LABEL_44:
      if ( v30 >= *(_DWORD *)(a1 + 40) )
        goto LABEL_20;
      *(_DWORD *)(a1 + 44) = v30 + 1;
      *v29 = a3;
      ++*(_QWORD *)(a1 + 16);
      goto LABEL_38;
    }
    v31 = 0;
    while ( a3 != *v22 )
    {
      if ( *v22 == -2 )
        v31 = v22;
      if ( v29 == ++v22 )
      {
        if ( !v31 )
          goto LABEL_44;
        *v31 = a3;
        --*(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 16);
        goto LABEL_38;
      }
    }
LABEL_21:
    v24 = sub_157F280(a3);
    v26 = v25;
    v27 = v24;
    if ( v24 != v25 )
    {
      while ( 1 )
      {
        sub_1A11B80(a1, (_DWORD *)v27);
        if ( !v27 )
          goto LABEL_29;
        v28 = *(_QWORD *)(v27 + 32);
        if ( !v28 )
          BUG();
        if ( *(_BYTE *)(v28 - 8) != 77 )
          break;
        v27 = v28 - 24;
        if ( v26 == v27 )
          return 1;
      }
      if ( v26 )
      {
        sub_1A11B80(a1, 0);
LABEL_29:
        BUG();
      }
    }
    return 1;
  }
  else
  {
LABEL_20:
    sub_16CCBA0(a1 + 16, a3);
    if ( !v23 )
      goto LABEL_21;
LABEL_38:
    v32 = *(unsigned int *)(a1 + 1880);
    if ( (unsigned int)v32 >= *(_DWORD *)(a1 + 1884) )
    {
      sub_16CD150(a1 + 1872, (const void *)(a1 + 1888), 0, 8, v20, v21);
      v32 = *(unsigned int *)(a1 + 1880);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1872) + 8 * v32) = a3;
    ++*(_DWORD *)(a1 + 1880);
    return 1;
  }
}
