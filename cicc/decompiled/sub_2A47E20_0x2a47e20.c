// Function: sub_2A47E20
// Address: 0x2a47e20
//
__int64 __fastcall sub_2A47E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v11; // r13
  int v12; // r14d
  __int64 v13; // rax
  __int64 i; // rdx
  unsigned int v15; // esi
  int v16; // r11d
  __int64 v17; // r8
  __int64 *v18; // rdx
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  int v22; // eax
  int v23; // ecx
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // r8
  int v29; // r10d
  __int64 *v30; // r9
  int v31; // eax
  int v32; // eax
  __int64 v33; // rdi
  __int64 *v34; // r8
  unsigned int v35; // r15d
  int v36; // r9d
  __int64 v37; // rsi

  v8 = *(unsigned int *)(a1 + 1608);
  v9 = *(_QWORD *)(a1 + 1592);
  if ( (_DWORD)v8 )
  {
    a4 = ((_DWORD)v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    a3 = v9 + 16 * a4;
    a5 = *(_QWORD *)a3;
    if ( a2 == *(_QWORD *)a3 )
    {
LABEL_3:
      if ( a3 != v9 + 16 * v8 )
        return *(_QWORD *)(a1 + 32) + 48LL * *(unsigned int *)(a3 + 8);
    }
    else
    {
      a3 = 1;
      while ( a5 != -4096 )
      {
        a6 = (unsigned int)(a3 + 1);
        a4 = ((_DWORD)v8 - 1) & (unsigned int)(a3 + a4);
        a3 = v9 + 16LL * (unsigned int)a4;
        a5 = *(_QWORD *)a3;
        if ( a2 == *(_QWORD *)a3 )
          goto LABEL_3;
        a3 = (unsigned int)a6;
      }
    }
  }
  v11 = *(unsigned int *)(a1 + 40);
  v12 = *(_DWORD *)(a1 + 40);
  if ( *(unsigned int *)(a1 + 44) < (unsigned __int64)(v11 + 1) )
    sub_2A45970(a1 + 32, v11 + 1, a3, a4, a5, a6);
  v13 = *(_QWORD *)(a1 + 32) + 48LL * *(unsigned int *)(a1 + 40);
  for ( i = *(_QWORD *)(a1 + 32) + 48 * (v11 + 1); i != v13; v13 += 48 )
  {
    if ( v13 )
    {
      *(_DWORD *)(v13 + 8) = 0;
      *(_QWORD *)v13 = v13 + 16;
      *(_DWORD *)(v13 + 12) = 4;
    }
  }
  v15 = *(_DWORD *)(a1 + 1608);
  *(_DWORD *)(a1 + 40) = v12 + 1;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 1584);
    goto LABEL_33;
  }
  v16 = 1;
  v17 = *(_QWORD *)(a1 + 1592);
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v17 + 16LL * v19);
  v21 = *v20;
  if ( a2 != *v20 )
  {
    while ( v21 != -4096 )
    {
      if ( !v18 && v21 == -8192 )
        v18 = v20;
      v19 = (v15 - 1) & (v16 + v19);
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( a2 == *v20 )
        goto LABEL_15;
      ++v16;
    }
    if ( !v18 )
      v18 = v20;
    v22 = *(_DWORD *)(a1 + 1600);
    ++*(_QWORD *)(a1 + 1584);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 1604) - v23 > v15 >> 3 )
      {
LABEL_29:
        *(_DWORD *)(a1 + 1600) = v23;
        if ( *v18 != -4096 )
          --*(_DWORD *)(a1 + 1604);
        *v18 = a2;
        *((_DWORD *)v18 + 2) = v12;
        return *(_QWORD *)(a1 + 32) + 48 * v11;
      }
      sub_D39D40(a1 + 1584, v15);
      v31 = *(_DWORD *)(a1 + 1608);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 1592);
        v34 = 0;
        v35 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v36 = 1;
        v23 = *(_DWORD *)(a1 + 1600) + 1;
        v18 = (__int64 *)(v33 + 16LL * v35);
        v37 = *v18;
        if ( a2 != *v18 )
        {
          while ( v37 != -4096 )
          {
            if ( !v34 && v37 == -8192 )
              v34 = v18;
            v35 = v32 & (v36 + v35);
            v18 = (__int64 *)(v33 + 16LL * v35);
            v37 = *v18;
            if ( a2 == *v18 )
              goto LABEL_29;
            ++v36;
          }
          if ( v34 )
            v18 = v34;
        }
        goto LABEL_29;
      }
LABEL_56:
      ++*(_DWORD *)(a1 + 1600);
      BUG();
    }
LABEL_33:
    sub_D39D40(a1 + 1584, 2 * v15);
    v24 = *(_DWORD *)(a1 + 1608);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 1592);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 1600) + 1;
      v18 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v18;
      if ( a2 != *v18 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v30 )
            v30 = v18;
          v27 = v25 & (v29 + v27);
          v18 = (__int64 *)(v26 + 16LL * v27);
          v28 = *v18;
          if ( a2 == *v18 )
            goto LABEL_29;
          ++v29;
        }
        if ( v30 )
          v18 = v30;
      }
      goto LABEL_29;
    }
    goto LABEL_56;
  }
LABEL_15:
  v11 = *((unsigned int *)v20 + 2);
  return *(_QWORD *)(a1 + 32) + 48 * v11;
}
