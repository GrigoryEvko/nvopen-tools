// Function: sub_2D7E430
// Address: 0x2d7e430
//
__int64 __fastcall sub_2D7E430(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // eax
  _QWORD *v10; // r9
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // r14
  char v15; // dl
  unsigned int v16; // esi
  unsigned int v17; // eax
  _QWORD *v18; // r8
  int v19; // ecx
  int v20; // r10d
  int v21; // eax
  __int64 v22; // rcx
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // edx
  __int64 v27; // rcx
  int v28; // edx
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // r9d
  _QWORD *v32; // rdi
  int v33; // r9d

  result = *a2;
  if ( *a2 != *a3 )
  {
    while ( 1 )
    {
      v13 = (_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
      if ( (result & 4) != 0 )
        v13 = (_QWORD *)*v13;
      v14 = v13[17];
      v15 = *(_BYTE *)(a1 + 8) & 1;
      if ( v15 )
      {
        v7 = a1 + 16;
        v8 = 3;
      }
      else
      {
        v16 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 16);
        if ( !v16 )
        {
          v17 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v18 = 0;
          v19 = (v17 >> 1) + 1;
          goto LABEL_16;
        }
        v8 = v16 - 1;
      }
      v9 = v8 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v10 = (_QWORD *)(v7 + 8LL * v9);
      v11 = *v10;
      if ( v14 != *v10 )
        break;
LABEL_5:
      v12 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*a2 & 4) != 0 || !v12 )
      {
        result = (v12 + 8) | 4;
        *a2 = result;
        if ( result == *a3 )
          return result;
      }
      else
      {
        *a2 = v12 + 144;
        result = v12 + 144;
        if ( result == *a3 )
          return result;
      }
    }
    v20 = 1;
    v18 = 0;
    while ( v11 != -4096 )
    {
      if ( v11 != -8192 || v18 )
        v10 = v18;
      v9 = v8 & (v20 + v9);
      v11 = *(_QWORD *)(v7 + 8LL * v9);
      if ( v14 == v11 )
        goto LABEL_5;
      ++v20;
      v18 = v10;
      v10 = (_QWORD *)(v7 + 8LL * v9);
    }
    v17 = *(_DWORD *)(a1 + 8);
    if ( !v18 )
      v18 = v10;
    ++*(_QWORD *)a1;
    v19 = (v17 >> 1) + 1;
    if ( v15 )
    {
      v16 = 4;
      if ( (unsigned int)(4 * v19) >= 0xC )
      {
LABEL_27:
        sub_2B75AF0(a1, 2 * v16);
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v22 = a1 + 16;
          v23 = 3;
        }
        else
        {
          v21 = *(_DWORD *)(a1 + 24);
          v22 = *(_QWORD *)(a1 + 16);
          if ( !v21 )
            goto LABEL_61;
          v23 = v21 - 1;
        }
        v24 = v23 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v18 = (_QWORD *)(v22 + 8LL * v24);
        v25 = *v18;
        if ( v14 == *v18 )
          goto LABEL_31;
        v33 = 1;
        v32 = 0;
        while ( v25 != -4096 )
        {
          if ( v25 == -8192 && !v32 )
            v32 = v18;
          v24 = v23 & (v33 + v24);
          v18 = (_QWORD *)(v22 + 8LL * v24);
          v25 = *v18;
          if ( v14 == *v18 )
            goto LABEL_31;
          ++v33;
        }
        goto LABEL_38;
      }
    }
    else
    {
      v16 = *(_DWORD *)(a1 + 24);
LABEL_16:
      if ( 4 * v19 >= 3 * v16 )
        goto LABEL_27;
    }
    if ( v16 - *(_DWORD *)(a1 + 12) - v19 > v16 >> 3 )
    {
LABEL_18:
      *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
      if ( *v18 != -4096 )
        --*(_DWORD *)(a1 + 12);
      *v18 = v14;
      goto LABEL_5;
    }
    sub_2B75AF0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v27 = a1 + 16;
      v28 = 3;
    }
    else
    {
      v26 = *(_DWORD *)(a1 + 24);
      v27 = *(_QWORD *)(a1 + 16);
      if ( !v26 )
      {
LABEL_61:
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        BUG();
      }
      v28 = v26 - 1;
    }
    v29 = v28 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v18 = (_QWORD *)(v27 + 8LL * v29);
    v30 = *v18;
    if ( v14 == *v18 )
    {
LABEL_31:
      v17 = *(_DWORD *)(a1 + 8);
      goto LABEL_18;
    }
    v31 = 1;
    v32 = 0;
    while ( v30 != -4096 )
    {
      if ( !v32 && v30 == -8192 )
        v32 = v18;
      v29 = v28 & (v31 + v29);
      v18 = (_QWORD *)(v27 + 8LL * v29);
      v30 = *v18;
      if ( v14 == *v18 )
        goto LABEL_31;
      ++v31;
    }
LABEL_38:
    if ( v32 )
      v18 = v32;
    goto LABEL_31;
  }
  return result;
}
