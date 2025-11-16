// Function: sub_1A54810
// Address: 0x1a54810
//
__int64 __fastcall sub_1A54810(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // r13
  unsigned int v5; // eax
  __int64 *v6; // r12
  int v7; // r15d
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rax
  bool v11; // zf
  __int64 *v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  _QWORD *v16; // r12
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  _QWORD *v21; // rdx
  _QWORD *v22; // rdi
  __int64 v23; // r8
  int v24; // edi
  int v25; // r14d
  _QWORD *v26; // r10
  unsigned int v27; // esi
  _QWORD *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rcx
  int v31; // edi
  __int64 v32; // r13
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v35; // r9
  int v36; // r10d
  int v37; // r14d
  _QWORD *v38; // r13
  unsigned int v39; // edi
  _QWORD *v40; // rcx
  __int64 v41; // r11
  __int64 v42; // rdx
  int v43; // ecx
  _BYTE v44[80]; // [rsp+10h] [rbp-50h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 1 )
  {
    if ( v4 )
      return result;
    v32 = *(unsigned int *)(a1 + 24);
    v6 = *(__int64 **)(a1 + 16);
    *(_BYTE *)(a1 + 8) = result | 1;
    v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
    *(_QWORD *)(a1 + 8) &= 1uLL;
    v12 = &v6[2 * v32];
    if ( v11 )
      goto LABEL_6;
    goto LABEL_38;
  }
  v5 = sub_1454B60(a2 - 1);
  v6 = *(__int64 **)(a1 + 16);
  v7 = v5;
  if ( v5 > 0x40 )
  {
    v9 = 2LL * v5;
    if ( !v4 )
    {
      v8 = *(unsigned int *)(a1 + 24);
LABEL_5:
      v10 = sub_22077B0(v9 * 8);
      v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v12 = &v6[2 * v8];
      *(_QWORD *)(a1 + 16) = v10;
      *(_DWORD *)(a1 + 24) = v7;
      if ( v11 )
      {
LABEL_6:
        v13 = *(_QWORD **)(a1 + 16);
        v14 = 2LL * *(unsigned int *)(a1 + 24);
LABEL_39:
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -8;
        }
        for ( j = v6; v12 != j; j += 2 )
        {
          v42 = *j;
          if ( *j != -16 && v42 != -8 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v35 = a1 + 16;
              v36 = 1;
            }
            else
            {
              v43 = *(_DWORD *)(a1 + 24);
              v35 = *(_QWORD *)(a1 + 16);
              if ( !v43 )
              {
                MEMORY[0] = *j;
                BUG();
              }
              v36 = v43 - 1;
            }
            v37 = 1;
            v38 = 0;
            v39 = v36 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
            v40 = (_QWORD *)(v35 + 16LL * v39);
            v41 = *v40;
            if ( *v40 != v42 )
            {
              while ( v41 != -8 )
              {
                if ( !v38 && v41 == -16 )
                  v38 = v40;
                v39 = v36 & (v37 + v39);
                v40 = (_QWORD *)(v35 + 16LL * v39);
                v41 = *v40;
                if ( v42 == *v40 )
                  goto LABEL_47;
                ++v37;
              }
              if ( v38 )
                v40 = v38;
            }
LABEL_47:
            *v40 = v42;
            v40[1] = j[1];
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return j___libc_free_0(v6);
      }
LABEL_38:
      v13 = (_QWORD *)(a1 + 16);
      v14 = 4;
      goto LABEL_39;
    }
  }
  else
  {
    if ( !v4 )
    {
      v8 = *(unsigned int *)(a1 + 24);
      v9 = 128;
      v7 = 64;
      goto LABEL_5;
    }
    v9 = 128;
    v7 = 64;
  }
  v15 = (__int64 *)(a1 + 16);
  v16 = v44;
  do
  {
    v17 = *v15;
    if ( *v15 != -8 && v17 != -16 )
    {
      if ( v16 )
        *v16 = v17;
      v16 += 2;
      *(v16 - 1) = v15[1];
    }
    v15 += 2;
  }
  while ( v15 != (__int64 *)(a1 + 48) );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v18 = (_QWORD *)sub_22077B0(v9 * 8);
  v19 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v18;
  v20 = v19 & 1;
  *(_QWORD *)(a1 + 8) = v20;
  if ( (_BYTE)v20 )
  {
    v18 = (_QWORD *)(a1 + 16);
    v9 = 4;
  }
  v21 = v18;
  v22 = &v18[v9];
  while ( 1 )
  {
    if ( v21 )
      *v18 = -8;
    v18 += 2;
    if ( v22 == v18 )
      break;
    v21 = v18;
  }
  for ( result = (__int64)v44; v16 != (_QWORD *)result; result += 16 )
  {
    v30 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -8 && v30 != -16 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v23 = a1 + 16;
        v24 = 1;
      }
      else
      {
        v31 = *(_DWORD *)(a1 + 24);
        v23 = *(_QWORD *)(a1 + 16);
        if ( !v31 )
        {
          MEMORY[0] = *(_QWORD *)result;
          BUG();
        }
        v24 = v31 - 1;
      }
      v25 = 1;
      v26 = 0;
      v27 = v24 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v28 = (_QWORD *)(v23 + 16LL * v27);
      v29 = *v28;
      if ( v30 != *v28 )
      {
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v26 )
            v26 = v28;
          v27 = v24 & (v25 + v27);
          v28 = (_QWORD *)(v23 + 16LL * v27);
          v29 = *v28;
          if ( v30 == *v28 )
            goto LABEL_28;
          ++v25;
        }
        if ( v26 )
          v28 = v26;
      }
LABEL_28:
      *v28 = v30;
      v28[1] = *(_QWORD *)(result + 8);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
