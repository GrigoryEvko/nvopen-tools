// Function: sub_2978150
// Address: 0x2978150
//
__int64 __fastcall sub_2978150(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r8d
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r10
  unsigned int v9; // r8d
  __int64 v10; // r13
  __int64 v11; // r9
  unsigned int v12; // edi
  int v13; // r10d
  unsigned int v14; // r12d
  unsigned int v15; // ecx
  unsigned int v16; // r15d
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  unsigned int v20; // eax
  char v21; // cl
  __int64 v22; // rax
  int v24; // eax
  int v25; // r11d
  int v26; // r10d
  __int64 *v27; // rsi
  __int64 v28; // rdi
  int v29; // eax
  int v30; // edx
  int v31; // eax
  int v32; // eax
  __int64 v33; // rdi
  unsigned int v34; // r12d
  __int64 v35; // rcx
  int v36; // r9d
  __int64 *v37; // r8
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdi
  int v41; // r9d
  unsigned int v42; // r12d
  __int64 v43; // rcx

  v4 = *(_DWORD *)(a1 + 136);
  v5 = *(_QWORD *)(a1 + 120);
  if ( v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a3 == *v7 )
      goto LABEL_3;
    v24 = 1;
    while ( v8 != -4096 )
    {
      v25 = v24 + 1;
      v6 = (v4 - 1) & (v24 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a3 == *v7 )
        goto LABEL_3;
      v24 = v25;
    }
  }
  v7 = (__int64 *)(v5 + 16LL * v4);
LABEL_3:
  v9 = *(_DWORD *)(a1 + 80);
  v10 = v7[1];
  v11 = *(_QWORD *)(a1 + 64);
  if ( v9 )
  {
    v12 = v9 - 1;
    v13 = 1;
    v14 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v15 = (v9 - 1) & v14;
    v16 = v15;
    v17 = (__int64 *)(v11 + 16LL * v15);
    v18 = *v17;
    v19 = *v17;
    if ( a2 == *v17 )
    {
LABEL_5:
      v20 = *((_DWORD *)v17 + 2);
      v21 = v20 & 0x3F;
      v22 = 8LL * (v20 >> 6);
      return (*(_QWORD *)(*(_QWORD *)(v10 + 96) + v22) >> v21) & 1LL;
    }
    while ( 1 )
    {
      if ( v19 == -4096 )
        return 0;
      v16 = v12 & (v16 + v13);
      v19 = *(_QWORD *)(v11 + 16LL * v16);
      if ( a2 == v19 )
        break;
      ++v13;
    }
    v26 = 1;
    v27 = 0;
    while ( v18 != -4096 )
    {
      if ( !v27 && v18 == -8192 )
        v27 = v17;
      v15 = v12 & (v26 + v15);
      v17 = (__int64 *)(v11 + 16LL * v15);
      v18 = *v17;
      if ( v19 == *v17 )
        goto LABEL_5;
      ++v26;
    }
    v28 = a1 + 56;
    if ( !v27 )
      v27 = v17;
    v29 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) >= 3 * v9 )
    {
      sub_CE2410(v28, 2 * v9);
      v31 = *(_DWORD *)(a1 + 80);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 64);
        v34 = v32 & v14;
        v30 = *(_DWORD *)(a1 + 72) + 1;
        v27 = (__int64 *)(v33 + 16LL * v34);
        v35 = *v27;
        if ( v19 == *v27 )
          goto LABEL_22;
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v37 )
            v37 = v27;
          v34 = v32 & (v36 + v34);
          v27 = (__int64 *)(v33 + 16LL * v34);
          v35 = *v27;
          if ( v19 == *v27 )
            goto LABEL_22;
          ++v36;
        }
LABEL_29:
        if ( v37 )
          v27 = v37;
        goto LABEL_22;
      }
    }
    else
    {
      if ( v9 - *(_DWORD *)(a1 + 76) - v30 > v9 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 72) = v30;
        if ( *v27 != -4096 )
          --*(_DWORD *)(a1 + 76);
        *v27 = v19;
        v21 = 0;
        v22 = 0;
        *((_DWORD *)v27 + 2) = 0;
        return (*(_QWORD *)(*(_QWORD *)(v10 + 96) + v22) >> v21) & 1LL;
      }
      sub_CE2410(v28, v9);
      v38 = *(_DWORD *)(a1 + 80);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 64);
        v41 = 1;
        v42 = v39 & v14;
        v37 = 0;
        v30 = *(_DWORD *)(a1 + 72) + 1;
        v27 = (__int64 *)(v40 + 16LL * v42);
        v43 = *v27;
        if ( *v27 == v19 )
          goto LABEL_22;
        while ( v43 != -4096 )
        {
          if ( !v37 && v43 == -8192 )
            v37 = v27;
          v42 = v39 & (v41 + v42);
          v27 = (__int64 *)(v40 + 16LL * v42);
          v43 = *v27;
          if ( v19 == *v27 )
            goto LABEL_22;
          ++v41;
        }
        goto LABEL_29;
      }
    }
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
  return 0;
}
