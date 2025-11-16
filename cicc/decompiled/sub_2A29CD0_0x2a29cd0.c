// Function: sub_2A29CD0
// Address: 0x2a29cd0
//
void __fastcall sub_2A29CD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 *v9; // r14
  __int64 v10; // rax
  unsigned int v11; // edx
  __int64 *v12; // r13
  __int64 v13; // rdi
  unsigned int v14; // esi
  __int64 v15; // rcx
  int v16; // r15d
  _QWORD *v17; // rdi
  __int64 v18; // r9
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  unsigned int v29; // ecx
  __int64 *v30; // rdx
  __int64 v31; // r9
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rcx
  int v37; // r8d
  __int64 v38; // rsi
  unsigned int v39; // eax
  int v40; // edx
  __int64 v41; // r9
  __int64 v42; // rax
  int v43; // eax
  int v44; // eax
  __int64 v45; // rcx
  int v46; // r8d
  __int64 v47; // rsi
  _QWORD *v48; // r10
  int v49; // r11d
  unsigned int v50; // eax
  __int64 v51; // r9
  int v52; // r9d
  int v53; // edx
  int v54; // r10d
  int v55; // r11d
  __int64 v56; // [rsp-40h] [rbp-40h] BYREF

  if ( !(_BYTE)qword_500A928 )
    return;
  v6 = sub_AA48A0(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v7 = *(_QWORD *)(a3 - 32);
  v8 = *(_QWORD *)(a1 + 192);
  v9 = (__int64 *)v6;
  v10 = *(unsigned int *)(a1 + 208);
  if ( !(_DWORD)v10 )
    return;
  v11 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( v7 == *v12 )
  {
LABEL_4:
    if ( v12 == (__int64 *)(v8 + 16 * v10) )
      return;
    v14 = *(_DWORD *)(a1 + 240);
    if ( v14 )
    {
      v15 = v12[1];
      v16 = 1;
      v17 = 0;
      v18 = *(_QWORD *)(a1 + 224);
      v19 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v20 = (_QWORD *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v15 == *v20 )
      {
LABEL_7:
        v22 = v20[1];
        goto LABEL_8;
      }
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v17 )
          v17 = v20;
        v19 = (v14 - 1) & (v16 + v19);
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v15 == *v20 )
          goto LABEL_7;
        ++v16;
      }
      if ( !v17 )
        v17 = v20;
      v43 = *(_DWORD *)(a1 + 232);
      ++*(_QWORD *)(a1 + 216);
      v40 = v43 + 1;
      if ( 4 * (v43 + 1) < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a1 + 236) - v40 > v14 >> 3 )
          goto LABEL_20;
        sub_2A29AF0(a1 + 216, v14);
        v44 = *(_DWORD *)(a1 + 240);
        if ( v44 )
        {
          v45 = v12[1];
          v46 = v44 - 1;
          v47 = *(_QWORD *)(a1 + 224);
          v48 = 0;
          v49 = 1;
          v50 = (v44 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v40 = *(_DWORD *)(a1 + 232) + 1;
          v17 = (_QWORD *)(v47 + 16LL * v50);
          v51 = *v17;
          if ( v45 != *v17 )
          {
            while ( v51 != -4096 )
            {
              if ( v51 == -8192 && !v48 )
                v48 = v17;
              v50 = v46 & (v49 + v50);
              v17 = (_QWORD *)(v47 + 16LL * v50);
              v51 = *v17;
              if ( v45 == *v17 )
                goto LABEL_20;
              ++v49;
            }
LABEL_36:
            if ( v48 )
              v17 = v48;
            goto LABEL_20;
          }
          goto LABEL_20;
        }
        goto LABEL_59;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 216);
    }
    sub_2A29AF0(a1 + 216, 2 * v14);
    v35 = *(_DWORD *)(a1 + 240);
    if ( v35 )
    {
      v36 = v12[1];
      v37 = v35 - 1;
      v38 = *(_QWORD *)(a1 + 224);
      v39 = (v35 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v40 = *(_DWORD *)(a1 + 232) + 1;
      v17 = (_QWORD *)(v38 + 16LL * v39);
      v41 = *v17;
      if ( *v17 != v36 )
      {
        v55 = 1;
        v48 = 0;
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v48 )
            v48 = v17;
          v39 = v37 & (v55 + v39);
          v17 = (_QWORD *)(v38 + 16LL * v39);
          v41 = *v17;
          if ( v36 == *v17 )
            goto LABEL_20;
          ++v55;
        }
        goto LABEL_36;
      }
LABEL_20:
      *(_DWORD *)(a1 + 232) = v40;
      if ( *v17 != -4096 )
        --*(_DWORD *)(a1 + 236);
      v42 = v12[1];
      v17[1] = 0;
      *v17 = v42;
      v22 = 0;
LABEL_8:
      v56 = v22;
      v23 = 0;
      v24 = sub_B9C770(v9, &v56, (__int64 *)1, 0, 1);
      if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
        v23 = sub_B91C10(a2, 7);
      v25 = sub_BA72D0(v23, v24);
      sub_B99FD0(a2, 7u, v25);
      v26 = *(unsigned int *)(a1 + 272);
      v27 = v12[1];
      v28 = *(_QWORD *)(a1 + 256);
      if ( (_DWORD)v26 )
      {
        v29 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v30 = (__int64 *)(v28 + 16LL * v29);
        v31 = *v30;
        if ( v27 == *v30 )
        {
LABEL_12:
          if ( v30 != (__int64 *)(v28 + 16 * v26) )
          {
            v32 = v30[1];
            v33 = 0;
            if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
              v33 = sub_B91C10(a2, 8);
            v34 = sub_BA72D0(v33, v32);
            sub_B99FD0(a2, 8u, v34);
          }
        }
        else
        {
          v53 = 1;
          while ( v31 != -4096 )
          {
            v54 = v53 + 1;
            v29 = (v26 - 1) & (v53 + v29);
            v30 = (__int64 *)(v28 + 16LL * v29);
            v31 = *v30;
            if ( v27 == *v30 )
              goto LABEL_12;
            v53 = v54;
          }
        }
      }
      return;
    }
LABEL_59:
    ++*(_DWORD *)(a1 + 232);
    BUG();
  }
  v52 = 1;
  while ( v13 != -4096 )
  {
    v11 = (v10 - 1) & (v52 + v11);
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v7 == *v12 )
      goto LABEL_4;
    ++v52;
  }
}
