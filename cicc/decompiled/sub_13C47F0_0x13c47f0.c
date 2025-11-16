// Function: sub_13C47F0
// Address: 0x13c47f0
//
unsigned __int64 __fastcall sub_13C47F0(unsigned __int64 *a1, __int64 a2, char a3)
{
  char v4; // r13
  unsigned __int64 v6; // rbx
  char v7; // cl
  unsigned __int64 v8; // rdi
  int v9; // esi
  unsigned int v10; // edx
  unsigned __int64 result; // rax
  __int64 v12; // r8
  unsigned int v13; // esi
  unsigned int v14; // edx
  int v15; // edi
  unsigned int v16; // r8d
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // rax
  int v20; // r10d
  unsigned __int64 v21; // r9
  unsigned __int64 v22; // rdi
  int v23; // ecx
  unsigned int v24; // edx
  __int64 v25; // rsi
  unsigned __int64 v26; // rsi
  int v27; // ecx
  unsigned int v28; // edx
  __int64 v29; // rdi
  int v30; // r9d
  unsigned __int64 v31; // r8
  int v32; // ecx
  int v33; // ecx
  unsigned __int64 v34; // r11
  int v35; // r9d

  v4 = a3;
  v6 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
  {
    v17 = (_QWORD *)sub_22077B0(272);
    v6 = (unsigned __int64)v17;
    if ( v17 )
    {
      *v17 = 0;
      v18 = v17 + 34;
      v19 = v17 + 2;
      *(v19 - 1) = 1;
      do
      {
        if ( v19 )
          *v19 = -8;
        v19 += 2;
      }
      while ( v19 != v18 );
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
    v13 = *(_DWORD *)(v6 + 24);
    v8 = *(_QWORD *)(v6 + 16);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(v6 + 8);
      ++*(_QWORD *)v6;
      result = 0;
      v15 = (v14 >> 1) + 1;
LABEL_10:
      v16 = 3 * v13;
      goto LABEL_11;
    }
    v9 = v13 - 1;
  }
  v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v8 + 16LL * v10;
  v12 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
  {
    v4 = a3 | *(_BYTE *)(result + 8);
    goto LABEL_6;
  }
  v20 = 1;
  v21 = 0;
  while ( v12 != -8 )
  {
    if ( v12 != -16 || v21 )
      result = v21;
    v10 = v9 & (v20 + v10);
    v34 = v8 + 16LL * v10;
    v12 = *(_QWORD *)v34;
    if ( a2 == *(_QWORD *)v34 )
    {
      result = v8 + 16LL * v10;
      v4 = a3 | *(_BYTE *)(v34 + 8);
      goto LABEL_6;
    }
    ++v20;
    v21 = result;
    result = v8 + 16LL * v10;
  }
  v14 = *(_DWORD *)(v6 + 8);
  v16 = 48;
  v13 = 16;
  if ( v21 )
    result = v21;
  ++*(_QWORD *)v6;
  v15 = (v14 >> 1) + 1;
  if ( !v7 )
  {
    v13 = *(_DWORD *)(v6 + 24);
    goto LABEL_10;
  }
LABEL_11:
  if ( 4 * v15 >= v16 )
  {
    sub_13C4410(v6, 2 * v13);
    if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
    {
      v22 = v6 + 16;
      v23 = 15;
    }
    else
    {
      v32 = *(_DWORD *)(v6 + 24);
      v22 = *(_QWORD *)(v6 + 16);
      if ( !v32 )
        goto LABEL_61;
      v23 = v32 - 1;
    }
    v24 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v22 + 16LL * v24;
    v25 = *(_QWORD *)result;
    if ( a2 != *(_QWORD *)result )
    {
      v35 = 1;
      v31 = 0;
      while ( v25 != -8 )
      {
        if ( !v31 && v25 == -16 )
          v31 = result;
        v24 = v23 & (v35 + v24);
        result = v22 + 16LL * v24;
        v25 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_31;
        ++v35;
      }
      goto LABEL_37;
    }
LABEL_31:
    v14 = *(_DWORD *)(v6 + 8);
    goto LABEL_13;
  }
  if ( v13 - *(_DWORD *)(v6 + 12) - v15 <= v13 >> 3 )
  {
    sub_13C4410(v6, v13);
    if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
    {
      v26 = v6 + 16;
      v27 = 15;
      goto LABEL_34;
    }
    v33 = *(_DWORD *)(v6 + 24);
    v26 = *(_QWORD *)(v6 + 16);
    if ( v33 )
    {
      v27 = v33 - 1;
LABEL_34:
      v28 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = v26 + 16LL * v28;
      v29 = *(_QWORD *)result;
      if ( a2 != *(_QWORD *)result )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v31 )
            v31 = result;
          v28 = v27 & (v30 + v28);
          result = v26 + 16LL * v28;
          v29 = *(_QWORD *)result;
          if ( a2 == *(_QWORD *)result )
            goto LABEL_31;
          ++v30;
        }
LABEL_37:
        if ( v31 )
          result = v31;
        goto LABEL_31;
      }
      goto LABEL_31;
    }
LABEL_61:
    *(_DWORD *)(v6 + 8) = (2 * (*(_DWORD *)(v6 + 8) >> 1) + 2) | *(_DWORD *)(v6 + 8) & 1;
    BUG();
  }
LABEL_13:
  *(_DWORD *)(v6 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(v6 + 12);
  *(_QWORD *)result = a2;
  *(_BYTE *)(result + 8) = 0;
LABEL_6:
  *(_BYTE *)(result + 8) = v4;
  return result;
}
