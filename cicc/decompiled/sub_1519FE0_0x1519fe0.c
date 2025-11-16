// Function: sub_1519FE0
// Address: 0x1519fe0
//
__int64 __fastcall sub_1519FE0(__int64 a1, _BYTE *a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // edx
  int v6; // edi
  _QWORD *v7; // rsi
  _BYTE *v8; // rcx
  __int64 result; // rax
  char v10; // al
  _QWORD *v11; // rdx
  unsigned int v12; // ecx
  unsigned int v13; // esi
  __int64 v14; // r8
  __int64 v15; // rdi
  _QWORD *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdi
  int v21; // eax
  unsigned int v22; // esi
  int v23; // eax
  int v24; // r8d
  unsigned int v25; // edx
  int v26; // ecx
  unsigned int v27; // edi
  int v28; // r9d
  __int64 v29; // rdi
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rax
  _BYTE *v33; // rcx
  __int64 v34; // rdi
  int v35; // eax
  int v36; // r8d
  __int64 v37; // rax
  __int64 v38; // rcx
  int v39; // esi
  _QWORD *v40; // rdx
  int v41; // esi

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
    v21 = *(_DWORD *)(a1 + 144);
    v7 = *(_QWORD **)(a1 + 136);
    if ( !v21 )
      goto LABEL_7;
    v6 = v21 - 1;
    v5 = (v21 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
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
    v23 = 1;
    while ( v8 != (_BYTE *)-8LL )
    {
      v24 = v23 + 1;
      v5 = v6 & (v23 + v5);
      v4 = &v7[2 * v5];
      v8 = (_BYTE *)*v4;
      if ( a2 == (_BYTE *)*v4 )
        goto LABEL_6;
      v23 = v24;
    }
  }
LABEL_7:
  v10 = *(_BYTE *)(a1 + 96) & 1;
  if ( v10 )
  {
    v11 = (_QWORD *)(a1 + 104);
    v12 = 0;
    v13 = 0;
    v14 = a1 + 104;
  }
  else
  {
    v22 = *(_DWORD *)(a1 + 112);
    v14 = *(_QWORD *)(a1 + 104);
    if ( !v22 )
    {
      v25 = *(_DWORD *)(a1 + 96);
      ++*(_QWORD *)(a1 + 88);
      v16 = 0;
      v26 = (v25 >> 1) + 1;
LABEL_26:
      v27 = 3 * v22;
      goto LABEL_27;
    }
    v13 = v22 - 1;
    v12 = v13 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (_QWORD *)(v14 + 16LL * v12);
  }
  v15 = *v11;
  if ( a2 == (_BYTE *)*v11 )
  {
LABEL_10:
    result = v11[1];
    if ( result )
      return result;
    v16 = v11;
    goto LABEL_12;
  }
  v28 = 1;
  v16 = 0;
  while ( v15 != -8 )
  {
    if ( !v16 && v15 == -16 )
      v16 = v11;
    v12 = v13 & (v28 + v12);
    v11 = (_QWORD *)(v14 + 16LL * v12);
    v15 = *v11;
    if ( a2 == (_BYTE *)*v11 )
      goto LABEL_10;
    ++v28;
  }
  v27 = 3;
  v22 = 1;
  if ( !v16 )
    v16 = v11;
  v25 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 88);
  v26 = (v25 >> 1) + 1;
  if ( !v10 )
  {
    v22 = *(_DWORD *)(a1 + 112);
    goto LABEL_26;
  }
LABEL_27:
  if ( 4 * v26 >= v27 )
  {
    sub_1519BD0(a1 + 88, 2 * v22);
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v29 = a1 + 104;
      v30 = 1;
    }
    else
    {
      v30 = *(_DWORD *)(a1 + 112);
      v29 = *(_QWORD *)(a1 + 104);
      if ( !v30 )
        goto LABEL_70;
    }
    v31 = v30 - 1;
    v32 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (_QWORD *)(v29 + 16 * v32);
    v33 = (_BYTE *)*v16;
    if ( a2 != (_BYTE *)*v16 )
    {
      v41 = 1;
      v40 = 0;
      while ( v33 != (_BYTE *)-8LL )
      {
        if ( !v40 && v33 == (_BYTE *)-16LL )
          v40 = v16;
        LODWORD(v32) = v31 & (v41 + v32);
        v16 = (_QWORD *)(v29 + 16LL * (unsigned int)v32);
        v33 = (_BYTE *)*v16;
        if ( a2 == (_BYTE *)*v16 )
          goto LABEL_41;
        ++v41;
      }
      goto LABEL_47;
    }
LABEL_41:
    v25 = *(_DWORD *)(a1 + 96);
    goto LABEL_29;
  }
  if ( v22 - *(_DWORD *)(a1 + 100) - v26 <= v22 >> 3 )
  {
    sub_1519BD0(a1 + 88, v22);
    if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
    {
      v34 = a1 + 104;
      v35 = 1;
      goto LABEL_44;
    }
    v35 = *(_DWORD *)(a1 + 112);
    v34 = *(_QWORD *)(a1 + 104);
    if ( v35 )
    {
LABEL_44:
      v36 = v35 - 1;
      v37 = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (_QWORD *)(v34 + 16 * v37);
      v38 = *v16;
      if ( a2 != (_BYTE *)*v16 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( !v40 && v38 == -16 )
            v40 = v16;
          LODWORD(v37) = v36 & (v39 + v37);
          v16 = (_QWORD *)(v34 + 16LL * (unsigned int)v37);
          v38 = *v16;
          if ( a2 == (_BYTE *)*v16 )
            goto LABEL_41;
          ++v39;
        }
LABEL_47:
        if ( v40 )
          v16 = v40;
        goto LABEL_41;
      }
      goto LABEL_41;
    }
LABEL_70:
    *(_DWORD *)(a1 + 96) = (2 * (*(_DWORD *)(a1 + 96) >> 1) + 2) | *(_DWORD *)(a1 + 96) & 1;
    BUG();
  }
LABEL_29:
  *(_DWORD *)(a1 + 96) = (2 * (v25 >> 1) + 2) | v25 & 1;
  if ( *v16 != -8 )
    --*(_DWORD *)(a1 + 100);
  *v16 = a2;
  v16[1] = 0;
LABEL_12:
  result = sub_1627350(*(_QWORD *)(a1 + 216), 0, 0, 2, 1);
  v20 = v16[1];
  v16[1] = result;
  if ( v20 )
  {
    sub_16307F0(v20, 0, v17, v18, v19);
    return v16[1];
  }
  return result;
}
