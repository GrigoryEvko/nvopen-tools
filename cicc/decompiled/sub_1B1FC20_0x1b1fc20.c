// Function: sub_1B1FC20
// Address: 0x1b1fc20
//
void __fastcall sub_1B1FC20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  int v10; // r9d
  unsigned int v11; // edx
  __int64 *v12; // r13
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rcx
  __int64 v16; // r9
  unsigned int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rdi
  __int64 v32; // r13
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rcx
  int v36; // r8d
  __int64 v37; // rsi
  unsigned int v38; // eax
  int v39; // edx
  _QWORD *v40; // rdi
  __int64 v41; // r9
  __int64 v42; // rax
  int v43; // r15d
  int v44; // eax
  int v45; // eax
  __int64 v46; // rcx
  int v47; // r8d
  __int64 v48; // rsi
  _QWORD *v49; // r10
  int v50; // r11d
  unsigned int v51; // eax
  __int64 v52; // r9
  int v53; // edx
  int v54; // r10d
  int v55; // edi
  __int64 *v56; // r15
  int v57; // r11d
  __int64 v58; // [rsp-40h] [rbp-40h] BYREF

  if ( !byte_4FB6BE0 )
    return;
  v6 = (__int64 *)sub_157E9C0(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v7 = *(unsigned int *)(a1 + 416);
  if ( !(_DWORD)v7 )
    return;
  v8 = *(_QWORD *)(a3 - 24);
  v9 = *(_QWORD *)(a1 + 400);
  v10 = 1;
  v11 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v8 != *v12 )
  {
    while ( 1 )
    {
      if ( v13 == -8 )
        return;
      v11 = (v7 - 1) & (v10 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        break;
      ++v10;
    }
  }
  if ( v12 == (__int64 *)(v9 + 16 * v7) )
    return;
  v14 = *(_DWORD *)(a1 + 448);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 424);
    goto LABEL_20;
  }
  v15 = v12[1];
  v16 = *(_QWORD *)(a1 + 432);
  v17 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v18 = (_QWORD *)(v16 + 16LL * v17);
  v19 = *v18;
  if ( v15 == *v18 )
  {
    v20 = v18[1];
    goto LABEL_8;
  }
  v43 = 1;
  v40 = 0;
  while ( 1 )
  {
    if ( v19 == -8 )
    {
      if ( !v40 )
        v40 = v18;
      v44 = *(_DWORD *)(a1 + 440);
      ++*(_QWORD *)(a1 + 424);
      v39 = v44 + 1;
      if ( 4 * (v44 + 1) < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 444) - v39 > v14 >> 3 )
          goto LABEL_22;
        sub_1B1FA60(a1 + 424, v14);
        v45 = *(_DWORD *)(a1 + 448);
        if ( v45 )
        {
          v46 = v12[1];
          v47 = v45 - 1;
          v48 = *(_QWORD *)(a1 + 432);
          v49 = 0;
          v50 = 1;
          v51 = (v45 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v39 = *(_DWORD *)(a1 + 440) + 1;
          v40 = (_QWORD *)(v48 + 16LL * v51);
          v52 = *v40;
          if ( *v40 != v46 )
          {
            while ( v52 != -8 )
            {
              if ( !v49 && v52 == -16 )
                v49 = v40;
              v51 = v47 & (v50 + v51);
              v40 = (_QWORD *)(v48 + 16LL * v51);
              v52 = *v40;
              if ( v46 == *v40 )
                goto LABEL_22;
              ++v50;
            }
LABEL_39:
            if ( v49 )
              v40 = v49;
            goto LABEL_22;
          }
          goto LABEL_22;
        }
        goto LABEL_64;
      }
LABEL_20:
      sub_1B1FA60(a1 + 424, 2 * v14);
      v34 = *(_DWORD *)(a1 + 448);
      if ( v34 )
      {
        v35 = v12[1];
        v36 = v34 - 1;
        v37 = *(_QWORD *)(a1 + 432);
        v38 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v39 = *(_DWORD *)(a1 + 440) + 1;
        v40 = (_QWORD *)(v37 + 16LL * v38);
        v41 = *v40;
        if ( *v40 != v35 )
        {
          v57 = 1;
          v49 = 0;
          while ( v41 != -8 )
          {
            if ( !v49 && v41 == -16 )
              v49 = v40;
            v38 = v36 & (v57 + v38);
            v40 = (_QWORD *)(v37 + 16LL * v38);
            v41 = *v40;
            if ( v35 == *v40 )
              goto LABEL_22;
            ++v57;
          }
          goto LABEL_39;
        }
LABEL_22:
        *(_DWORD *)(a1 + 440) = v39;
        if ( *v40 != -8 )
          --*(_DWORD *)(a1 + 444);
        v42 = v12[1];
        v40[1] = 0;
        *v40 = v42;
        v20 = 0;
        goto LABEL_8;
      }
LABEL_64:
      ++*(_DWORD *)(a1 + 440);
      BUG();
    }
    if ( v19 != -16 || v40 )
      v18 = v40;
    v55 = v43 + 1;
    v17 = (v14 - 1) & (v43 + v17);
    v56 = (__int64 *)(v16 + 16LL * v17);
    v19 = *v56;
    if ( v15 == *v56 )
      break;
    v43 = v55;
    v40 = v18;
    v18 = (_QWORD *)(v16 + 16LL * v17);
  }
  v20 = v56[1];
LABEL_8:
  v58 = v20;
  v21 = sub_1627350(v6, &v58, (__int64 *)1, 0, 1);
  v22 = *(_QWORD *)(a2 + 48);
  v23 = v21;
  if ( v22 || *(__int16 *)(a2 + 18) < 0 )
    v22 = sub_1625790(a2, 7);
  v24 = sub_1631960(v22, v23);
  sub_1625C10(a2, 7, v24);
  v25 = *(unsigned int *)(a1 + 480);
  if ( (_DWORD)v25 )
  {
    v26 = v12[1];
    v27 = *(_QWORD *)(a1 + 464);
    v28 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v29 = (__int64 *)(v27 + 16LL * v28);
    v30 = *v29;
    if ( v26 == *v29 )
    {
LABEL_12:
      if ( v29 != (__int64 *)(v27 + 16 * v25) )
      {
        v31 = *(_QWORD *)(a2 + 48);
        v32 = v29[1];
        if ( v31 || *(__int16 *)(a2 + 18) < 0 )
          v31 = sub_1625790(a2, 8);
        v33 = sub_1631960(v31, v32);
        sub_1625C10(a2, 8, v33);
      }
    }
    else
    {
      v53 = 1;
      while ( v30 != -8 )
      {
        v54 = v53 + 1;
        v28 = (v25 - 1) & (v53 + v28);
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( v26 == *v29 )
          goto LABEL_12;
        v53 = v54;
      }
    }
  }
}
