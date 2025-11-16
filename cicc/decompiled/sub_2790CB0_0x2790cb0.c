// Function: sub_2790CB0
// Address: 0x2790cb0
//
unsigned __int64 __fastcall sub_2790CB0(__int64 a1, _BYTE *a2, int a3)
{
  unsigned int v5; // esi
  __int64 v6; // r9
  unsigned int v7; // r8d
  unsigned __int64 result; // rax
  _BYTE *v9; // rdi
  unsigned int v10; // esi
  int v11; // r15d
  _DWORD *v12; // r11
  int v13; // r13d
  __int64 v14; // r8
  unsigned int v15; // edi
  _DWORD *v16; // rax
  int v17; // ecx
  int v18; // r11d
  _DWORD *v19; // rcx
  int v20; // eax
  int v21; // edi
  int v22; // eax
  int v23; // ecx
  int v24; // eax
  int v25; // esi
  __int64 v26; // r9
  _BYTE *v27; // r8
  int v28; // r11d
  _DWORD *v29; // r10
  int v30; // eax
  __int64 v31; // r8
  _DWORD *v32; // r9
  unsigned int v33; // r13d
  int v34; // r10d
  _BYTE *v35; // rsi
  int v36; // eax
  int v37; // edi
  __int64 v38; // r8
  unsigned int v39; // esi
  int v40; // eax
  int v41; // r10d
  _DWORD *v42; // r9
  int v43; // eax
  int v44; // esi
  __int64 v45; // rdi
  _DWORD *v46; // r8
  unsigned int v47; // r13d
  int v48; // r9d
  int v49; // eax
  int v50; // [rsp+Ch] [rbp-34h]
  int v51; // [rsp+Ch] [rbp-34h]
  int v52; // [rsp+Ch] [rbp-34h]
  int v53; // [rsp+Ch] [rbp-34h]

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_31;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v6 + 16LL * v7;
  v9 = *(_BYTE **)result;
  if ( a2 == *(_BYTE **)result )
    goto LABEL_3;
  v18 = 1;
  v19 = 0;
  while ( v9 != (_BYTE *)-4096LL )
  {
    if ( v19 || v9 != (_BYTE *)-8192LL )
      result = (unsigned __int64)v19;
    v7 = (v5 - 1) & (v18 + v7);
    v9 = *(_BYTE **)(v6 + 16LL * v7);
    if ( a2 == v9 )
      goto LABEL_3;
    ++v18;
    v19 = (_DWORD *)result;
    result = v6 + 16LL * v7;
  }
  if ( !v19 )
    v19 = (_DWORD *)result;
  v20 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v5 )
  {
LABEL_31:
    v50 = a3;
    sub_D39D40(a1, 2 * v5);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      result = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v50;
      v19 = (_DWORD *)(v26 + 16 * result);
      v27 = *(_BYTE **)v19;
      if ( a2 != *(_BYTE **)v19 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != (_BYTE *)-4096LL )
        {
          if ( v27 == (_BYTE *)-8192LL && !v29 )
            v29 = v19;
          result = v25 & (unsigned int)(v28 + result);
          v19 = (_DWORD *)(v26 + 16LL * (unsigned int)result);
          v27 = *(_BYTE **)v19;
          if ( a2 == *(_BYTE **)v19 )
            goto LABEL_14;
          ++v28;
        }
        if ( v29 )
          v19 = v29;
      }
      goto LABEL_14;
    }
    goto LABEL_84;
  }
  result = v5 - *(_DWORD *)(a1 + 20) - v21;
  if ( (unsigned int)result <= v5 >> 3 )
  {
    v51 = a3;
    sub_D39D40(a1, v5);
    v30 = *(_DWORD *)(a1 + 24);
    if ( v30 )
    {
      result = (unsigned int)(v30 - 1);
      v31 = *(_QWORD *)(a1 + 8);
      v32 = 0;
      v33 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = 1;
      v21 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v51;
      v19 = (_DWORD *)(v31 + 16LL * v33);
      v35 = *(_BYTE **)v19;
      if ( a2 != *(_BYTE **)v19 )
      {
        while ( v35 != (_BYTE *)-4096LL )
        {
          if ( v35 == (_BYTE *)-8192LL && !v32 )
            v32 = v19;
          v33 = result & (v34 + v33);
          v19 = (_DWORD *)(v31 + 16LL * v33);
          v35 = *(_BYTE **)v19;
          if ( a2 == *(_BYTE **)v19 )
            goto LABEL_14;
          ++v34;
        }
        if ( v32 )
          v19 = v32;
      }
      goto LABEL_14;
    }
LABEL_84:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v21;
  if ( *(_QWORD *)v19 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v19 = a2;
  v19[2] = a3;
LABEL_3:
  if ( *a2 != 84 )
    return result;
  v10 = *(_DWORD *)(a1 + 144);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_45;
  }
  v11 = 1;
  v12 = 0;
  v13 = 37 * a3;
  v14 = *(_QWORD *)(a1 + 128);
  v15 = (v10 - 1) & (37 * a3);
  v16 = (_DWORD *)(v14 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a3 )
  {
    while ( v17 != -1 )
    {
      if ( !v12 && v17 == -2 )
        v12 = v16;
      v15 = (v10 - 1) & (v11 + v15);
      v16 = (_DWORD *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a3 )
        goto LABEL_6;
      ++v11;
    }
    if ( !v12 )
      v12 = v16;
    v22 = *(_DWORD *)(a1 + 136);
    ++*(_QWORD *)(a1 + 120);
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 140) - v23 > v10 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a1 + 136) = v23;
        if ( *v12 != -1 )
          --*(_DWORD *)(a1 + 140);
        *v12 = a3;
        result = (unsigned __int64)(v12 + 2);
        *((_QWORD *)v12 + 1) = 0;
        goto LABEL_7;
      }
      v53 = a3;
      sub_27908B0(a1 + 120, v10);
      v43 = *(_DWORD *)(a1 + 144);
      if ( v43 )
      {
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 128);
        a3 = v53;
        v46 = 0;
        v47 = (v43 - 1) & v13;
        v48 = 1;
        v23 = *(_DWORD *)(a1 + 136) + 1;
        v12 = (_DWORD *)(v45 + 16LL * v47);
        v49 = *v12;
        if ( *v12 != v53 )
        {
          while ( v49 != -1 )
          {
            if ( !v46 && v49 == -2 )
              v46 = v12;
            v47 = v44 & (v48 + v47);
            v12 = (_DWORD *)(v45 + 16LL * v47);
            v49 = *v12;
            if ( *v12 == v53 )
              goto LABEL_27;
            ++v48;
          }
          if ( v46 )
            v12 = v46;
        }
        goto LABEL_27;
      }
LABEL_83:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_45:
    v52 = a3;
    sub_27908B0(a1 + 120, 2 * v10);
    v36 = *(_DWORD *)(a1 + 144);
    if ( v36 )
    {
      a3 = v52;
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 128);
      v23 = *(_DWORD *)(a1 + 136) + 1;
      v39 = (v36 - 1) & (37 * v52);
      v12 = (_DWORD *)(v38 + 16LL * v39);
      v40 = *v12;
      if ( *v12 != v52 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -1 )
        {
          if ( !v42 && v40 == -2 )
            v42 = v12;
          v39 = v37 & (v41 + v39);
          v12 = (_DWORD *)(v38 + 16LL * v39);
          v40 = *v12;
          if ( *v12 == v52 )
            goto LABEL_27;
          ++v41;
        }
        if ( v42 )
          v12 = v42;
      }
      goto LABEL_27;
    }
    goto LABEL_83;
  }
LABEL_6:
  result = (unsigned __int64)(v16 + 2);
LABEL_7:
  *(_QWORD *)result = a2;
  return result;
}
