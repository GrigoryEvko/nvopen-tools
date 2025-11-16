// Function: sub_A05F80
// Address: 0xa05f80
//
__int64 __fastcall sub_A05F80(__int64 a1, _BYTE *a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // edx
  int v6; // edi
  _QWORD *v7; // rsi
  _BYTE *v8; // rcx
  __int64 result; // rax
  char v10; // cl
  unsigned int v11; // esi
  __int64 v12; // r9
  unsigned int v13; // esi
  unsigned int v14; // edi
  _QWORD *v15; // rax
  _BYTE *v16; // r8
  __int64 *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdi
  int v23; // eax
  int v24; // eax
  int v25; // r8d
  unsigned int v26; // eax
  _QWORD *v27; // rdx
  int v28; // edi
  unsigned int v29; // r8d
  int v30; // r10d
  unsigned int v31; // eax
  int v32; // esi
  __int64 v33; // r9
  _BYTE *v34; // rcx
  unsigned int v35; // eax
  int v36; // esi
  __int64 v37; // r9
  __int64 v38; // rcx
  int v39; // r8d
  _QWORD *v40; // rdi
  int v41; // esi
  int v42; // esi
  int v43; // r8d

  if ( !a2 )
    return 0;
  if ( *a2 )
    return (__int64)a2;
  if ( (*(_BYTE *)(a1 + 128) & 1) != 0 )
  {
    v4 = (_QWORD *)(a1 + 136);
    v5 = 0;
    v6 = 0;
    v7 = v4;
  }
  else
  {
    v23 = *(_DWORD *)(a1 + 144);
    v7 = *(_QWORD **)(a1 + 136);
    if ( !v23 )
      goto LABEL_7;
    v6 = v23 - 1;
    v5 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v4 = &v7[2 * v5];
  }
  v8 = (_BYTE *)*v4;
  if ( a2 == (_BYTE *)*v4 )
  {
LABEL_6:
    result = v4[1];
    if ( result )
      return result;
  }
  else
  {
    v24 = 1;
    while ( v8 != (_BYTE *)-4096LL )
    {
      v25 = v24 + 1;
      v5 = v6 & (v24 + v5);
      v4 = &v7[2 * v5];
      v8 = (_BYTE *)*v4;
      if ( a2 == (_BYTE *)*v4 )
        goto LABEL_6;
      v24 = v25;
    }
  }
LABEL_7:
  v10 = *(_BYTE *)(a1 + 96) & 1;
  if ( v10 )
  {
    v15 = (_QWORD *)(a1 + 104);
    v14 = 0;
    v13 = 0;
    v12 = a1 + 104;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 112);
    v12 = *(_QWORD *)(a1 + 104);
    if ( !v11 )
    {
      v26 = *(_DWORD *)(a1 + 96);
      ++*(_QWORD *)(a1 + 88);
      v27 = 0;
      v28 = (v26 >> 1) + 1;
LABEL_25:
      v29 = 3 * v11;
      goto LABEL_26;
    }
    v13 = v11 - 1;
    v14 = v13 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = (_QWORD *)(v12 + 16LL * v14);
  }
  v16 = (_BYTE *)*v15;
  if ( a2 == (_BYTE *)*v15 )
  {
LABEL_11:
    v17 = v15 + 1;
    result = v15[1];
    if ( result )
      return result;
    goto LABEL_12;
  }
  v30 = 1;
  v27 = 0;
  while ( v16 != (_BYTE *)-4096LL )
  {
    if ( !v27 && v16 == (_BYTE *)-8192LL )
      v27 = v15;
    v14 = v13 & (v30 + v14);
    v15 = (_QWORD *)(v12 + 16LL * v14);
    v16 = (_BYTE *)*v15;
    if ( a2 == (_BYTE *)*v15 )
      goto LABEL_11;
    ++v30;
  }
  v29 = 3;
  v11 = 1;
  if ( !v27 )
    v27 = v15;
  v26 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 88);
  v28 = (v26 >> 1) + 1;
  if ( !v10 )
  {
    v11 = *(_DWORD *)(a1 + 112);
    goto LABEL_25;
  }
LABEL_26:
  if ( 4 * v28 >= v29 )
  {
    sub_A05AA0(a1 + 88, 2 * v11);
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v27 = (_QWORD *)(a1 + 104);
      v31 = 0;
      v32 = 0;
      v33 = a1 + 104;
    }
    else
    {
      v41 = *(_DWORD *)(a1 + 112);
      v33 = *(_QWORD *)(a1 + 104);
      if ( !v41 )
        goto LABEL_69;
      v32 = v41 - 1;
      v31 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = (_QWORD *)(v33 + 16LL * v31);
    }
    v34 = (_BYTE *)*v27;
    if ( a2 != (_BYTE *)*v27 )
    {
      v43 = 1;
      v40 = 0;
      while ( v34 != (_BYTE *)-4096LL )
      {
        if ( !v40 && v34 == (_BYTE *)-8192LL )
          v40 = v27;
        v31 = v32 & (v43 + v31);
        v27 = (_QWORD *)(v33 + 16LL * v31);
        v34 = (_BYTE *)*v27;
        if ( a2 == (_BYTE *)*v27 )
          goto LABEL_40;
        ++v43;
      }
      goto LABEL_46;
    }
LABEL_40:
    v26 = *(_DWORD *)(a1 + 96);
    goto LABEL_28;
  }
  if ( v11 - *(_DWORD *)(a1 + 100) - v28 <= v11 >> 3 )
  {
    sub_A05AA0(a1 + 88, v11);
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v27 = (_QWORD *)(a1 + 104);
      v35 = 0;
      v36 = 0;
      v37 = a1 + 104;
      goto LABEL_43;
    }
    v42 = *(_DWORD *)(a1 + 112);
    v37 = *(_QWORD *)(a1 + 104);
    if ( v42 )
    {
      v36 = v42 - 1;
      v35 = v36 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
      v27 = (_QWORD *)(v37 + 16LL * v35);
LABEL_43:
      v38 = *v27;
      if ( a2 != (_BYTE *)*v27 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v40 )
            v40 = v27;
          v35 = v36 & (v39 + v35);
          v27 = (_QWORD *)(v37 + 16LL * v35);
          v38 = *v27;
          if ( a2 == (_BYTE *)*v27 )
            goto LABEL_40;
          ++v39;
        }
LABEL_46:
        if ( v40 )
          v27 = v40;
        goto LABEL_40;
      }
      goto LABEL_40;
    }
LABEL_69:
    *(_DWORD *)(a1 + 96) = (2 * (*(_DWORD *)(a1 + 96) >> 1) + 2) | *(_DWORD *)(a1 + 96) & 1;
    BUG();
  }
LABEL_28:
  *(_DWORD *)(a1 + 96) = (2 * (v26 >> 1) + 2) | v26 & 1;
  if ( *v27 != -4096 )
    --*(_DWORD *)(a1 + 100);
  *v27 = a2;
  v17 = v27 + 1;
  v27[1] = 0;
LABEL_12:
  result = sub_B9C770(*(_QWORD *)(a1 + 216), 0, 0, 2, 1);
  v22 = *v17;
  *v17 = result;
  if ( v22 )
  {
    sub_BA65D0(v22, 0, v18, v19, v20, v21);
    return *v17;
  }
  return result;
}
