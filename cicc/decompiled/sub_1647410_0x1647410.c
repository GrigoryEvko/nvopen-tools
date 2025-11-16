// Function: sub_1647410
// Address: 0x1647410
//
void __fastcall sub_1647410(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r10d
  unsigned int v8; // edi
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rcx
  int v12; // edi
  int v13; // ecx
  _QWORD *v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rbx
  bool v17; // zf
  _BYTE *v18; // rsi
  _QWORD *v19; // r14
  _QWORD *v20; // rbx
  __int64 v21; // r8
  _QWORD *v22; // r10
  int v23; // r11d
  unsigned int v24; // eax
  _QWORD *v25; // rdi
  __int64 v26; // rcx
  unsigned int v27; // esi
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // r8
  unsigned int v31; // edx
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // rax
  int v35; // eax
  int v36; // ecx
  int v37; // ecx
  __int64 v38; // r8
  _QWORD *v39; // r9
  int v40; // r11d
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r11d
  int v44; // r9d
  int v45; // r9d
  __int64 v46; // r10
  unsigned int v47; // edx
  __int64 v48; // rdi
  int v49; // r8d
  _QWORD *v50; // rsi
  int v51; // esi
  int v52; // esi
  __int64 v53; // r9
  int v54; // r8d
  unsigned int v55; // r14d
  _QWORD *v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v59; // [rsp+30h] [rbp-60h] BYREF
  __int64 v60; // [rsp+38h] [rbp-58h]
  _QWORD v61[10]; // [rsp+40h] [rbp-50h] BYREF

  v2 = a1 + 64;
  v5 = *(_DWORD *)(a1 + 88);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_65;
  }
  v6 = *(_QWORD *)(a1 + 72);
  v7 = 1;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v6 + 8LL * v8);
  v10 = 0;
  v11 = *v9;
  if ( *v9 == a2 )
    return;
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v10 )
      v10 = v9;
    v8 = (v5 - 1) & (v7 + v8);
    v9 = (_QWORD *)(v6 + 8LL * v8);
    v11 = *v9;
    if ( *v9 == a2 )
      return;
    ++v7;
  }
  v12 = *(_DWORD *)(a1 + 80);
  if ( !v10 )
    v10 = v9;
  ++*(_QWORD *)(a1 + 64);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_65:
    sub_1647260(v2, 2 * v5);
    v44 = *(_DWORD *)(a1 + 88);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 72);
      v47 = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 80) + 1;
      v10 = (_QWORD *)(v46 + 8LL * v47);
      v48 = *v10;
      if ( *v10 != a2 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -8 )
        {
          if ( v48 == -16 && !v50 )
            v50 = v10;
          v47 = v45 & (v49 + v47);
          v10 = (_QWORD *)(v46 + 8LL * v47);
          v48 = *v10;
          if ( *v10 == a2 )
            goto LABEL_14;
          ++v49;
        }
        if ( v50 )
          v10 = v50;
      }
      goto LABEL_14;
    }
    goto LABEL_94;
  }
  if ( v5 - *(_DWORD *)(a1 + 84) - v13 <= v5 >> 3 )
  {
    sub_1647260(v2, v5);
    v51 = *(_DWORD *)(a1 + 88);
    if ( v51 )
    {
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 72);
      v54 = 1;
      v55 = v52 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 80) + 1;
      v56 = 0;
      v10 = (_QWORD *)(v53 + 8LL * v55);
      v57 = *v10;
      if ( *v10 != a2 )
      {
        while ( v57 != -8 )
        {
          if ( !v56 && v57 == -16 )
            v56 = v10;
          v55 = v52 & (v54 + v55);
          v10 = (_QWORD *)(v53 + 8LL * v55);
          v57 = *v10;
          if ( *v10 == a2 )
            goto LABEL_14;
          ++v54;
        }
        if ( v56 )
          v10 = v56;
      }
      goto LABEL_14;
    }
LABEL_94:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 80) = v13;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 84);
  *v10 = a2;
  v14 = v61;
  v59 = v61;
  v61[0] = a2;
  v60 = 0x400000001LL;
  v15 = 1;
  do
  {
    v16 = v14[v15 - 1];
    LODWORD(v60) = v15 - 1;
    if ( *(_BYTE *)(v16 + 8) == 13 )
    {
      v17 = *(_BYTE *)(a1 + 120) == 0;
      v58 = v16;
      if ( v17 || *(_QWORD *)(v16 + 24) )
      {
        v18 = *(_BYTE **)(a1 + 104);
        if ( v18 == *(_BYTE **)(a1 + 112) )
        {
          sub_14EFD20(a1 + 96, v18, &v58);
        }
        else
        {
          if ( v18 )
          {
            *(_QWORD *)v18 = v16;
            v18 = *(_BYTE **)(a1 + 104);
          }
          *(_QWORD *)(a1 + 104) = v18 + 8;
        }
      }
    }
    v19 = *(_QWORD **)(v16 + 16);
    v20 = &v19[*(unsigned int *)(v16 + 12)];
    if ( v19 != v20 )
    {
      while ( 1 )
      {
        v27 = *(_DWORD *)(a1 + 88);
        --v20;
        if ( !v27 )
          break;
        v21 = *(_QWORD *)(a1 + 72);
        v22 = 0;
        v23 = 1;
        v24 = (v27 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
        v25 = (_QWORD *)(v21 + 8LL * v24);
        v26 = *v25;
        if ( *v25 == *v20 )
        {
LABEL_27:
          if ( v19 == v20 )
            goto LABEL_37;
        }
        else
        {
          while ( v26 != -8 )
          {
            if ( v22 || v26 != -16 )
              v25 = v22;
            v24 = (v27 - 1) & (v23 + v24);
            v26 = *(_QWORD *)(v21 + 8LL * v24);
            if ( *v20 == v26 )
              goto LABEL_27;
            ++v23;
            v22 = v25;
            v25 = (_QWORD *)(v21 + 8LL * v24);
          }
          v35 = *(_DWORD *)(a1 + 80);
          if ( !v22 )
            v22 = v25;
          ++*(_QWORD *)(a1 + 64);
          v33 = v35 + 1;
          if ( 4 * v33 < 3 * v27 )
          {
            if ( v27 - *(_DWORD *)(a1 + 84) - v33 > v27 >> 3 )
              goto LABEL_32;
            sub_1647260(v2, v27);
            v36 = *(_DWORD *)(a1 + 88);
            if ( !v36 )
            {
LABEL_93:
              ++*(_DWORD *)(a1 + 80);
              BUG();
            }
            v37 = v36 - 1;
            v38 = *(_QWORD *)(a1 + 72);
            v39 = 0;
            v40 = 1;
            v41 = v37 & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
            v22 = (_QWORD *)(v38 + 8LL * v41);
            v42 = *v22;
            v33 = *(_DWORD *)(a1 + 80) + 1;
            if ( *v20 == *v22 )
              goto LABEL_32;
            while ( v42 != -8 )
            {
              if ( !v39 && v42 == -16 )
                v39 = v22;
              v41 = v37 & (v40 + v41);
              v22 = (_QWORD *)(v38 + 8LL * v41);
              v42 = *v22;
              if ( *v20 == *v22 )
                goto LABEL_32;
              ++v40;
            }
            goto LABEL_52;
          }
LABEL_30:
          sub_1647260(v2, 2 * v27);
          v28 = *(_DWORD *)(a1 + 88);
          if ( !v28 )
            goto LABEL_93;
          v29 = v28 - 1;
          v30 = *(_QWORD *)(a1 + 72);
          v31 = v29 & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
          v22 = (_QWORD *)(v30 + 8LL * v31);
          v32 = *v22;
          v33 = *(_DWORD *)(a1 + 80) + 1;
          if ( *v22 == *v20 )
            goto LABEL_32;
          v43 = 1;
          v39 = 0;
          while ( v32 != -8 )
          {
            if ( v32 == -16 && !v39 )
              v39 = v22;
            v31 = v29 & (v43 + v31);
            v22 = (_QWORD *)(v30 + 8LL * v31);
            v32 = *v22;
            if ( *v20 == *v22 )
              goto LABEL_32;
            ++v43;
          }
LABEL_52:
          if ( v39 )
            v22 = v39;
LABEL_32:
          *(_DWORD *)(a1 + 80) = v33;
          if ( *v22 != -8 )
            --*(_DWORD *)(a1 + 84);
          *v22 = *v20;
          v34 = (unsigned int)v60;
          if ( (unsigned int)v60 >= HIDWORD(v60) )
          {
            sub_16CD150(&v59, v61, 0, 8);
            v34 = (unsigned int)v60;
          }
          v59[v34] = *v20;
          LODWORD(v60) = v60 + 1;
          if ( v19 == v20 )
            goto LABEL_37;
        }
      }
      ++*(_QWORD *)(a1 + 64);
      goto LABEL_30;
    }
LABEL_37:
    v15 = v60;
    v14 = v59;
  }
  while ( (_DWORD)v60 );
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
}
