// Function: sub_3080DC0
// Address: 0x3080dc0
//
__int64 __fastcall sub_3080DC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  _BYTE *v3; // rsi
  __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v6; // rdx
  unsigned int v7; // esi
  int v8; // edi
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  _QWORD *v14; // r15
  __int64 v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // edx
  int v20; // r8d
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // r9
  _QWORD *v24; // rbx
  __int64 v25; // rax
  _QWORD *v26; // rcx
  __int64 v27; // r12
  __int64 v28; // r8
  unsigned int v29; // edi
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // r13
  unsigned int v33; // esi
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  __int64 v37; // rax
  _QWORD *v38; // r10
  int v39; // edx
  __int64 v40; // rdi
  __int64 v41; // rax
  int v42; // r11d
  int v43; // eax
  int v44; // eax
  int v45; // eax
  __int64 v46; // rdi
  _QWORD *v47; // r8
  unsigned int v48; // r14d
  int v49; // r9d
  __int64 v50; // rsi
  int i; // edx
  int v52; // r10d
  int v53; // r11d
  _QWORD *v54; // r9
  int v55; // r11d
  unsigned int v56; // r9d
  _QWORD *v57; // [rsp+0h] [rbp-40h]
  _QWORD *v58; // [rsp+0h] [rbp-40h]

  v2 = a1;
  v3 = *(_BYTE **)a1;
  if ( !**(_BYTE **)a1 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    if ( !*(_DWORD *)(v12 + 16) )
      goto LABEL_6;
    v13 = *(_QWORD **)(v12 + 8);
    v14 = &v13[2 * *(unsigned int *)(v12 + 24)];
    if ( v13 == v14 )
      goto LABEL_6;
    while ( 1 )
    {
      v15 = *v13;
      v16 = v13;
      if ( *v13 != -8192 && v15 != -4096 )
        break;
      v13 += 2;
      if ( v14 == v13 )
        goto LABEL_6;
    }
    if ( v14 == v13 )
    {
LABEL_6:
      *v3 = 1;
      goto LABEL_2;
    }
    while ( 1 )
    {
      v17 = *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL);
      v18 = *(_QWORD *)(v17 + 8);
      v19 = *(_DWORD *)(v17 + 24);
      if ( v19 )
      {
        v20 = v19 - 1;
        v21 = (v19 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v22 = (__int64 *)(v18 + 16LL * v21);
        v23 = *v22;
        if ( *v22 != v15 )
        {
          for ( i = 1; ; i = v52 )
          {
            if ( v23 == -4096 )
              goto LABEL_27;
            v52 = i + 1;
            v21 = v20 & (i + v21);
            v22 = (__int64 *)(v18 + 16LL * v21);
            v23 = *v22;
            if ( *v22 == v15 )
              break;
          }
        }
        v24 = (_QWORD *)v22[1];
        if ( v24 )
          break;
      }
LABEL_27:
      v16 += 2;
      if ( v16 != v14 )
      {
        while ( 1 )
        {
          v15 = *v16;
          if ( *v16 != -8192 && v15 != -4096 )
            break;
          v16 += 2;
          if ( v14 == v16 )
            goto LABEL_31;
        }
        if ( v14 != v16 )
          continue;
      }
LABEL_31:
      v3 = *(_BYTE **)v2;
      goto LABEL_6;
    }
    v25 = v2;
    v26 = v16;
    v27 = v25;
    while ( 1 )
    {
      v32 = *(_QWORD *)(v27 + 24);
      v33 = *(_DWORD *)(v32 + 24);
      if ( !v33 )
        break;
      v28 = *(_QWORD *)(v32 + 8);
      v29 = (v33 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v30 = (_QWORD *)(v28 + 8LL * v29);
      v31 = (_QWORD *)*v30;
      if ( v24 == (_QWORD *)*v30 )
      {
LABEL_18:
        v24 = (_QWORD *)*v24;
        if ( !v24 )
          goto LABEL_26;
      }
      else
      {
        v42 = 1;
        v38 = 0;
        while ( v31 != (_QWORD *)-4096LL )
        {
          if ( v31 != (_QWORD *)-8192LL || v38 )
            v30 = v38;
          v29 = (v33 - 1) & (v42 + v29);
          v31 = *(_QWORD **)(v28 + 8LL * v29);
          if ( v31 == v24 )
            goto LABEL_18;
          ++v42;
          v38 = v30;
          v30 = (_QWORD *)(v28 + 8LL * v29);
        }
        v43 = *(_DWORD *)(v32 + 16);
        if ( !v38 )
          v38 = v30;
        ++*(_QWORD *)v32;
        v39 = v43 + 1;
        if ( 4 * (v43 + 1) < 3 * v33 )
        {
          if ( v33 - *(_DWORD *)(v32 + 20) - v39 <= v33 >> 3 )
          {
            v58 = v26;
            sub_3080BF0(v32, v33);
            v44 = *(_DWORD *)(v32 + 24);
            if ( !v44 )
            {
LABEL_75:
              ++*(_DWORD *)(v32 + 16);
              BUG();
            }
            v45 = v44 - 1;
            v46 = *(_QWORD *)(v32 + 8);
            v47 = 0;
            v48 = v45 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v49 = 1;
            v38 = (_QWORD *)(v46 + 8LL * v48);
            v39 = *(_DWORD *)(v32 + 16) + 1;
            v26 = v58;
            v50 = *v38;
            if ( (_QWORD *)*v38 != v24 )
            {
              while ( v50 != -4096 )
              {
                if ( v50 == -8192 && !v47 )
                  v47 = v38;
                v55 = v49 + 1;
                v56 = v45 & (v48 + v49);
                v38 = (_QWORD *)(v46 + 8LL * v56);
                v48 = v56;
                v50 = *v38;
                if ( (_QWORD *)*v38 == v24 )
                  goto LABEL_23;
                v49 = v55;
              }
              if ( v47 )
                v38 = v47;
            }
          }
          goto LABEL_23;
        }
LABEL_21:
        v57 = v26;
        sub_3080BF0(v32, 2 * v33);
        v34 = *(_DWORD *)(v32 + 24);
        if ( !v34 )
          goto LABEL_75;
        v35 = v34 - 1;
        v36 = *(_QWORD *)(v32 + 8);
        LODWORD(v37) = (v34 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v38 = (_QWORD *)(v36 + 8LL * (unsigned int)v37);
        v39 = *(_DWORD *)(v32 + 16) + 1;
        v26 = v57;
        v40 = *v38;
        if ( (_QWORD *)*v38 != v24 )
        {
          v53 = 1;
          v54 = 0;
          while ( v40 != -4096 )
          {
            if ( !v54 && v40 == -8192 )
              v54 = v38;
            v37 = v35 & (unsigned int)(v37 + v53);
            v38 = (_QWORD *)(v36 + 8 * v37);
            v40 = *v38;
            if ( (_QWORD *)*v38 == v24 )
              goto LABEL_23;
            ++v53;
          }
          if ( v54 )
            v38 = v54;
        }
LABEL_23:
        *(_DWORD *)(v32 + 16) = v39;
        if ( *v38 != -4096 )
          --*(_DWORD *)(v32 + 20);
        *v38 = v24;
        v24 = (_QWORD *)*v24;
        if ( !v24 )
        {
LABEL_26:
          v41 = v27;
          v16 = v26;
          v2 = v41;
          goto LABEL_27;
        }
      }
    }
    ++*(_QWORD *)v32;
    goto LABEL_21;
  }
LABEL_2:
  v4 = *(_QWORD *)(v2 + 24);
  v5 = *(_DWORD *)(v4 + 24);
  v6 = *(_QWORD *)(v4 + 8);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = 1;
    v5 = 1;
    v9 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = *(_QWORD *)(v6 + 8LL * v9);
    if ( a2 != v10 )
    {
      while ( 1 )
      {
        if ( v10 == -4096 )
          return 0;
        v9 = v7 & (v8 + v9);
        v10 = *(_QWORD *)(v6 + 8LL * v9);
        if ( a2 == v10 )
          break;
        ++v8;
      }
      return 1;
    }
  }
  return v5;
}
