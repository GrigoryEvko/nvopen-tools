// Function: sub_941B90
// Address: 0x941b90
//
__int64 __fastcall sub_941B90(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // esi
  int v14; // r10d
  __int64 v15; // r8
  _QWORD *v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r12
  int v21; // r9d
  int v22; // eax
  int v23; // edx
  int v24; // eax
  int v25; // ecx
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // r10d
  _QWORD *v30; // r9
  int v31; // eax
  int v32; // eax
  __int64 v33; // rsi
  _QWORD *v34; // r8
  unsigned int v35; // r15d
  int v36; // r9d
  __int64 v37; // rcx

  v4 = *(unsigned int *)(a1 + 600);
  v5 = *(_QWORD *)(a1 + 584);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        v9 = v7[1];
        if ( v9 )
          return v9;
      }
    }
    else
    {
      v11 = 1;
      while ( v8 != -4096 )
      {
        v21 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v21;
      }
    }
  }
  v12 = sub_9430C0(a1, a2);
  v13 = *(_DWORD *)(a1 + 600);
  v9 = v12;
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 576);
    goto LABEL_30;
  }
  v14 = 1;
  v15 = *(_QWORD *)(a1 + 584);
  v16 = 0;
  v17 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (_QWORD *)(v15 + 16LL * v17);
  v19 = *v18;
  if ( *v18 != a2 )
  {
    while ( v19 != -4096 )
    {
      if ( !v16 && v19 == -8192 )
        v16 = v18;
      v17 = (v13 - 1) & (v14 + v17);
      v18 = (_QWORD *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( *v18 == a2 )
        goto LABEL_10;
      ++v14;
    }
    if ( !v16 )
      v16 = v18;
    v22 = *(_DWORD *)(a1 + 592);
    ++*(_QWORD *)(a1 + 576);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(a1 + 596) - v23 > v13 >> 3 )
      {
LABEL_26:
        *(_DWORD *)(a1 + 592) = v23;
        if ( *v16 != -4096 )
          --*(_DWORD *)(a1 + 596);
        *v16 = a2;
        v20 = v16 + 1;
        v16[1] = 0;
        goto LABEL_12;
      }
      sub_941970(a1 + 576, v13);
      v31 = *(_DWORD *)(a1 + 600);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 584);
        v34 = 0;
        v35 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v36 = 1;
        v23 = *(_DWORD *)(a1 + 592) + 1;
        v16 = (_QWORD *)(v33 + 16LL * v35);
        v37 = *v16;
        if ( *v16 != a2 )
        {
          while ( v37 != -4096 )
          {
            if ( !v34 && v37 == -8192 )
              v34 = v16;
            v35 = v32 & (v36 + v35);
            v16 = (_QWORD *)(v33 + 16LL * v35);
            v37 = *v16;
            if ( *v16 == a2 )
              goto LABEL_26;
            ++v36;
          }
          if ( v34 )
            v16 = v34;
        }
        goto LABEL_26;
      }
LABEL_53:
      ++*(_DWORD *)(a1 + 592);
      BUG();
    }
LABEL_30:
    sub_941970(a1 + 576, 2 * v13);
    v24 = *(_DWORD *)(a1 + 600);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 584);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 592) + 1;
      v16 = (_QWORD *)(v26 + 16LL * v27);
      v28 = *v16;
      if ( *v16 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v30 )
            v30 = v16;
          v27 = v25 & (v29 + v27);
          v16 = (_QWORD *)(v26 + 16LL * v27);
          v28 = *v16;
          if ( *v16 == a2 )
            goto LABEL_26;
          ++v29;
        }
        if ( v30 )
          v16 = v30;
      }
      goto LABEL_26;
    }
    goto LABEL_53;
  }
LABEL_10:
  v20 = v18 + 1;
  if ( v18[1] )
    sub_B91220(v18 + 1);
LABEL_12:
  *v20 = v9;
  if ( v9 )
    sub_B96E90(v20, v9, 1);
  return v9;
}
