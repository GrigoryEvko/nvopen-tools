// Function: sub_21EE750
// Address: 0x21ee750
//
__int64 __fastcall sub_21EE750(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rsi
  int v7; // edi
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rsi
  int v16; // edx
  __int64 v17; // r8
  int v18; // edi
  unsigned int v19; // esi
  __int64 *v20; // rdx
  __int64 v21; // r9
  _QWORD *v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // rcx
  __int64 v25; // r12
  __int64 v26; // r8
  unsigned int v27; // edi
  _QWORD *v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // r13
  unsigned int v31; // esi
  int v32; // eax
  int v33; // esi
  __int64 v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // r10
  int v37; // edx
  __int64 v38; // rdi
  __int64 v39; // rax
  int v40; // r11d
  int v41; // eax
  int v42; // eax
  int v43; // eax
  __int64 v44; // rdi
  _QWORD *v45; // r8
  unsigned int v46; // r14d
  int v47; // r9d
  __int64 v48; // rsi
  int i; // edx
  int v50; // r10d
  int v51; // r11d
  _QWORD *v52; // r9
  int v53; // r11d
  unsigned int v54; // r9d
  _QWORD *v55; // [rsp+0h] [rbp-40h]
  _QWORD *v56; // [rsp+0h] [rbp-40h]

  v2 = a1;
  v3 = *(_BYTE **)a1;
  if ( !**(_BYTE **)a1 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    if ( !*(_DWORD *)(v10 + 16) )
      goto LABEL_6;
    v11 = *(_QWORD **)(v10 + 8);
    v12 = &v11[2 * *(unsigned int *)(v10 + 24)];
    if ( v11 == v12 )
      goto LABEL_6;
    while ( 1 )
    {
      v13 = *v11;
      v14 = v11;
      if ( *v11 != -16 && v13 != -8 )
        break;
      v11 += 2;
      if ( v12 == v11 )
        goto LABEL_6;
    }
    if ( v12 == v11 )
    {
LABEL_6:
      *v3 = 1;
      goto LABEL_2;
    }
    while ( 1 )
    {
      v15 = *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL);
      v16 = *(_DWORD *)(v15 + 256);
      if ( v16 )
      {
        v17 = *(_QWORD *)(v15 + 240);
        v18 = v16 - 1;
        v19 = (v16 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( *v20 != v13 )
        {
          for ( i = 1; ; i = v50 )
          {
            if ( v21 == -8 )
              goto LABEL_27;
            v50 = i + 1;
            v19 = v18 & (i + v19);
            v20 = (__int64 *)(v17 + 16LL * v19);
            v21 = *v20;
            if ( *v20 == v13 )
              break;
          }
        }
        v22 = (_QWORD *)v20[1];
        if ( v22 )
          break;
      }
LABEL_27:
      v14 += 2;
      if ( v14 != v12 )
      {
        while ( 1 )
        {
          v13 = *v14;
          if ( *v14 != -16 && v13 != -8 )
            break;
          v14 += 2;
          if ( v12 == v14 )
            goto LABEL_31;
        }
        if ( v12 != v14 )
          continue;
      }
LABEL_31:
      v3 = *(_BYTE **)v2;
      goto LABEL_6;
    }
    v23 = v2;
    v24 = v14;
    v25 = v23;
    while ( 1 )
    {
      v30 = *(_QWORD *)(v25 + 24);
      v31 = *(_DWORD *)(v30 + 24);
      if ( !v31 )
        break;
      v26 = *(_QWORD *)(v30 + 8);
      v27 = (v31 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v28 = (_QWORD *)(v26 + 8LL * v27);
      v29 = (_QWORD *)*v28;
      if ( (_QWORD *)*v28 == v22 )
      {
LABEL_18:
        v22 = (_QWORD *)*v22;
        if ( !v22 )
          goto LABEL_26;
      }
      else
      {
        v40 = 1;
        v36 = 0;
        while ( v29 != (_QWORD *)-8LL )
        {
          if ( v36 || v29 != (_QWORD *)-16LL )
            v28 = v36;
          v27 = (v31 - 1) & (v40 + v27);
          v29 = *(_QWORD **)(v26 + 8LL * v27);
          if ( v29 == v22 )
            goto LABEL_18;
          ++v40;
          v36 = v28;
          v28 = (_QWORD *)(v26 + 8LL * v27);
        }
        v41 = *(_DWORD *)(v30 + 16);
        if ( !v36 )
          v36 = v28;
        ++*(_QWORD *)v30;
        v37 = v41 + 1;
        if ( 4 * (v41 + 1) < 3 * v31 )
        {
          if ( v31 - *(_DWORD *)(v30 + 20) - v37 <= v31 >> 3 )
          {
            v56 = v24;
            sub_21EE5A0(v30, v31);
            v42 = *(_DWORD *)(v30 + 24);
            if ( !v42 )
            {
LABEL_75:
              ++*(_DWORD *)(v30 + 16);
              BUG();
            }
            v43 = v42 - 1;
            v44 = *(_QWORD *)(v30 + 8);
            v45 = 0;
            v46 = v43 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v47 = 1;
            v36 = (_QWORD *)(v44 + 8LL * v46);
            v37 = *(_DWORD *)(v30 + 16) + 1;
            v24 = v56;
            v48 = *v36;
            if ( (_QWORD *)*v36 != v22 )
            {
              while ( v48 != -8 )
              {
                if ( v48 == -16 && !v45 )
                  v45 = v36;
                v53 = v47 + 1;
                v54 = v43 & (v46 + v47);
                v36 = (_QWORD *)(v44 + 8LL * v54);
                v46 = v54;
                v48 = *v36;
                if ( (_QWORD *)*v36 == v22 )
                  goto LABEL_23;
                v47 = v53;
              }
              if ( v45 )
                v36 = v45;
            }
          }
          goto LABEL_23;
        }
LABEL_21:
        v55 = v24;
        sub_21EE5A0(v30, 2 * v31);
        v32 = *(_DWORD *)(v30 + 24);
        if ( !v32 )
          goto LABEL_75;
        v33 = v32 - 1;
        v34 = *(_QWORD *)(v30 + 8);
        LODWORD(v35) = (v32 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v36 = (_QWORD *)(v34 + 8LL * (unsigned int)v35);
        v37 = *(_DWORD *)(v30 + 16) + 1;
        v24 = v55;
        v38 = *v36;
        if ( (_QWORD *)*v36 != v22 )
        {
          v51 = 1;
          v52 = 0;
          while ( v38 != -8 )
          {
            if ( !v52 && v38 == -16 )
              v52 = v36;
            v35 = v33 & (unsigned int)(v35 + v51);
            v36 = (_QWORD *)(v34 + 8 * v35);
            v38 = *v36;
            if ( (_QWORD *)*v36 == v22 )
              goto LABEL_23;
            ++v51;
          }
          if ( v52 )
            v36 = v52;
        }
LABEL_23:
        *(_DWORD *)(v30 + 16) = v37;
        if ( *v36 != -8 )
          --*(_DWORD *)(v30 + 20);
        *v36 = v22;
        v22 = (_QWORD *)*v22;
        if ( !v22 )
        {
LABEL_26:
          v39 = v25;
          v14 = v24;
          v2 = v39;
          goto LABEL_27;
        }
      }
    }
    ++*(_QWORD *)v30;
    goto LABEL_21;
  }
LABEL_2:
  v4 = *(_QWORD *)(v2 + 24);
  result = *(unsigned int *)(v4 + 24);
  if ( (_DWORD)result )
  {
    v6 = *(_QWORD *)(v4 + 8);
    v7 = result - 1;
    v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = *(_QWORD *)(v6 + 8LL * v8);
    result = 1;
    if ( a2 != v9 )
    {
      while ( 1 )
      {
        if ( v9 == -8 )
          return 0;
        v8 = v7 & (result + v8);
        v9 = *(_QWORD *)(v6 + 8LL * v8);
        if ( a2 == v9 )
          break;
        LODWORD(result) = result + 1;
      }
      return 1;
    }
  }
  return result;
}
