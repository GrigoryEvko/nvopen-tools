// Function: sub_D1E0D0
// Address: 0xd1e0d0
//
unsigned __int64 __fastcall sub_D1E0D0(unsigned __int64 *a1, __int64 a2, char a3)
{
  char v3; // r14
  unsigned __int64 v6; // rbx
  char v7; // di
  unsigned __int64 v8; // r8
  int v9; // esi
  unsigned int v10; // ecx
  unsigned __int64 result; // rax
  __int64 v12; // r9
  _BYTE *v13; // rdx
  unsigned int v14; // esi
  unsigned int v15; // eax
  _QWORD *v16; // rdx
  int v17; // ecx
  unsigned int v18; // r8d
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // rax
  int v22; // r11d
  unsigned __int64 v23; // rdi
  int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  int v28; // ecx
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r9d
  _QWORD *v32; // r8
  int v33; // ecx
  int v34; // ecx
  int v35; // r9d

  v3 = a3;
  v6 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
  {
    v19 = (_QWORD *)sub_22077B0(272);
    v6 = (unsigned __int64)v19;
    if ( v19 )
    {
      *v19 = 0;
      v20 = v19 + 34;
      v21 = v19 + 2;
      *(v21 - 1) = 1;
      do
      {
        if ( v21 )
          *v21 = -4096;
        v21 += 2;
      }
      while ( v21 != v20 );
    }
    *a1 = v6 | *a1 & 7;
  }
  v7 = *(_BYTE *)(v6 + 8) & 1;
  if ( v7 )
  {
    v8 = v6 + 16;
    v9 = 15;
  }
  else
  {
    v14 = *(_DWORD *)(v6 + 24);
    v8 = *(_QWORD *)(v6 + 16);
    if ( !v14 )
    {
      v15 = *(_DWORD *)(v6 + 8);
      ++*(_QWORD *)v6;
      v16 = 0;
      v17 = (v15 >> 1) + 1;
LABEL_10:
      v18 = 3 * v14;
      goto LABEL_11;
    }
    v9 = v14 - 1;
  }
  v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v8 + 16LL * v10;
  v12 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
  {
LABEL_5:
    v13 = (_BYTE *)(result + 8);
    v3 = a3 | *(_BYTE *)(result + 8);
    goto LABEL_6;
  }
  v22 = 1;
  v16 = 0;
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v16 )
      v16 = (_QWORD *)result;
    v10 = v9 & (v22 + v10);
    result = v8 + 16LL * v10;
    v12 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_5;
    ++v22;
  }
  v18 = 48;
  v14 = 16;
  if ( !v16 )
    v16 = (_QWORD *)result;
  v15 = *(_DWORD *)(v6 + 8);
  ++*(_QWORD *)v6;
  v17 = (v15 >> 1) + 1;
  if ( !v7 )
  {
    v14 = *(_DWORD *)(v6 + 24);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v17 >= v18 )
  {
    sub_D1DC90(v6, 2 * v14);
    if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
    {
      v23 = v6 + 16;
      v24 = 15;
    }
    else
    {
      v33 = *(_DWORD *)(v6 + 24);
      v23 = *(_QWORD *)(v6 + 16);
      if ( !v33 )
        goto LABEL_60;
      v24 = v33 - 1;
    }
    v25 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (_QWORD *)(v23 + 16LL * v25);
    v26 = *v16;
    if ( a2 != *v16 )
    {
      v35 = 1;
      v32 = 0;
      while ( v26 != -4096 )
      {
        if ( !v32 && v26 == -8192 )
          v32 = v16;
        v25 = v24 & (v35 + v25);
        v16 = (_QWORD *)(v23 + 16LL * v25);
        v26 = *v16;
        if ( a2 == *v16 )
          goto LABEL_31;
        ++v35;
      }
      goto LABEL_37;
    }
LABEL_31:
    v15 = *(_DWORD *)(v6 + 8);
    goto LABEL_13;
  }
  if ( v14 - *(_DWORD *)(v6 + 12) - v17 <= v14 >> 3 )
  {
    sub_D1DC90(v6, v14);
    if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
    {
      v27 = v6 + 16;
      v28 = 15;
      goto LABEL_34;
    }
    v34 = *(_DWORD *)(v6 + 24);
    v27 = *(_QWORD *)(v6 + 16);
    if ( v34 )
    {
      v28 = v34 - 1;
LABEL_34:
      v29 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (_QWORD *)(v27 + 16LL * v29);
      v30 = *v16;
      if ( a2 != *v16 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v32 )
            v32 = v16;
          v29 = v28 & (v31 + v29);
          v16 = (_QWORD *)(v27 + 16LL * v29);
          v30 = *v16;
          if ( a2 == *v16 )
            goto LABEL_31;
          ++v31;
        }
LABEL_37:
        if ( v32 )
          v16 = v32;
        goto LABEL_31;
      }
      goto LABEL_31;
    }
LABEL_60:
    *(_DWORD *)(v6 + 8) = (2 * (*(_DWORD *)(v6 + 8) >> 1) + 2) | *(_DWORD *)(v6 + 8) & 1;
    BUG();
  }
LABEL_13:
  result = (2 * (v15 >> 1) + 2) | v15 & 1;
  *(_DWORD *)(v6 + 8) = result;
  if ( *v16 != -4096 )
    --*(_DWORD *)(v6 + 12);
  *v16 = a2;
  v13 = v16 + 1;
  *v13 = 0;
LABEL_6:
  *v13 = v3;
  return result;
}
