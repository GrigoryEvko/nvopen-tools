// Function: sub_1BEB4C0
// Address: 0x1beb4c0
//
__int64 __fastcall sub_1BEB4C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rsi
  unsigned int v22; // r8d
  __int64 *v23; // rcx
  int v24; // r9d
  int v25; // r11d
  _QWORD *v26; // r10
  int v27; // ecx
  int v28; // ecx
  int v29; // eax
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // edx
  __int64 v33; // rdi
  int v34; // r10d
  _QWORD *v35; // r9
  int v36; // eax
  int v37; // edx
  __int64 v38; // r8
  int v39; // r9d
  unsigned int v40; // r14d
  _QWORD *v41; // rdi
  __int64 v42; // rsi

  v4 = *(unsigned int *)(a1 + 104);
  if ( !(_DWORD)v4 )
    goto LABEL_8;
  v5 = *(_QWORD *)(a1 + 88);
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( *v7 != a2 )
  {
    v11 = 1;
    while ( v8 != -8 )
    {
      v24 = v11 + 1;
      v6 = (v4 - 1) & (v11 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == a2 )
        goto LABEL_3;
      v11 = v24;
    }
LABEL_8:
    v12 = sub_22077B0(40);
    v9 = v12;
    if ( v12 )
    {
      *(_BYTE *)v12 = 0;
      *(_QWORD *)(v12 + 8) = v12 + 24;
      *(_QWORD *)(v12 + 16) = 0x100000000LL;
      *(_QWORD *)(v12 + 32) = a2;
    }
    v13 = *(_QWORD *)(a1 + 16);
    v14 = *(__int64 **)(v13 + 120);
    if ( *(__int64 **)(v13 + 128) != v14 )
      goto LABEL_11;
    v21 = &v14[*(unsigned int *)(v13 + 140)];
    v22 = *(_DWORD *)(v13 + 140);
    if ( v14 == v21 )
    {
LABEL_25:
      if ( v22 >= *(_DWORD *)(v13 + 136) )
      {
LABEL_11:
        sub_16CCBA0(v13 + 112, v9);
        goto LABEL_12;
      }
      *(_DWORD *)(v13 + 140) = v22 + 1;
      *v21 = v9;
      ++*(_QWORD *)(v13 + 112);
    }
    else
    {
      v23 = 0;
      while ( v9 != *v14 )
      {
        if ( *v14 == -2 )
          v23 = v14;
        if ( v21 == ++v14 )
        {
          if ( !v23 )
            goto LABEL_25;
          *v23 = v9;
          --*(_DWORD *)(v13 + 144);
          ++*(_QWORD *)(v13 + 112);
          break;
        }
      }
    }
LABEL_12:
    v15 = *(_DWORD *)(a1 + 104);
    v16 = a1 + 80;
    if ( v15 )
    {
      v17 = *(_QWORD *)(a1 + 88);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (_QWORD *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == a2 )
      {
LABEL_14:
        v19[1] = v9;
        return v9;
      }
      v25 = 1;
      v26 = 0;
      while ( v20 != -8 )
      {
        if ( !v26 && v20 == -16 )
          v26 = v19;
        v18 = (v15 - 1) & (v25 + v18);
        v19 = (_QWORD *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == a2 )
          goto LABEL_14;
        ++v25;
      }
      v27 = *(_DWORD *)(a1 + 96);
      if ( v26 )
        v19 = v26;
      ++*(_QWORD *)(a1 + 80);
      v28 = v27 + 1;
      if ( 4 * v28 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 100) - v28 > v15 >> 3 )
        {
LABEL_33:
          *(_DWORD *)(a1 + 96) = v28;
          if ( *v19 != -8 )
            --*(_DWORD *)(a1 + 100);
          *v19 = a2;
          v19[1] = 0;
          goto LABEL_14;
        }
        sub_1BA21E0(v16, v15);
        v36 = *(_DWORD *)(a1 + 104);
        if ( v36 )
        {
          v37 = v36 - 1;
          v38 = *(_QWORD *)(a1 + 88);
          v39 = 1;
          v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v28 = *(_DWORD *)(a1 + 96) + 1;
          v41 = 0;
          v19 = (_QWORD *)(v38 + 16LL * v40);
          v42 = *v19;
          if ( *v19 != a2 )
          {
            while ( v42 != -8 )
            {
              if ( !v41 && v42 == -16 )
                v41 = v19;
              v40 = v37 & (v39 + v40);
              v19 = (_QWORD *)(v38 + 16LL * v40);
              v42 = *v19;
              if ( *v19 == a2 )
                goto LABEL_33;
              ++v39;
            }
            if ( v41 )
              v19 = v41;
          }
          goto LABEL_33;
        }
LABEL_65:
        ++*(_DWORD *)(a1 + 96);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 80);
    }
    sub_1BA21E0(v16, 2 * v15);
    v29 = *(_DWORD *)(a1 + 104);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 88);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = *(_DWORD *)(a1 + 96) + 1;
      v19 = (_QWORD *)(v31 + 16LL * v32);
      v33 = *v19;
      if ( *v19 != a2 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v35 )
            v35 = v19;
          v32 = v30 & (v34 + v32);
          v19 = (_QWORD *)(v31 + 16LL * v32);
          v33 = *v19;
          if ( *v19 == a2 )
            goto LABEL_33;
          ++v34;
        }
        if ( v35 )
          v19 = v35;
      }
      goto LABEL_33;
    }
    goto LABEL_65;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v4) )
    goto LABEL_8;
  return v7[1];
}
