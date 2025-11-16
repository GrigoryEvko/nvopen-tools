// Function: sub_2F96BB0
// Address: 0x2f96bb0
//
_QWORD *__fastcall sub_2F96BB0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  _QWORD *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  _QWORD *j; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r10
  int v20; // esi
  int v21; // r15d
  __int64 *v22; // r14
  unsigned int v23; // r8d
  __int64 *v24; // rcx
  __int64 v25; // r11
  __int64 v26; // rdx
  _QWORD *result; // rax
  int v28; // esi
  _QWORD *v29; // r14
  _QWORD *v30; // rsi
  _QWORD *v31; // rax
  _QWORD *v32; // r13
  __int64 v33; // rcx
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *k; // rdx
  __int64 v38; // rcx
  _QWORD *v39; // r9
  int v40; // edi
  int v41; // r11d
  __int64 *v42; // r10
  unsigned int v43; // esi
  __int64 *v44; // rcx
  __int64 v45; // r8
  __int64 v46; // rdx
  int v47; // ecx
  _BYTE v48[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v29 = (_QWORD *)(a1 + 16);
    v30 = (_QWORD *)(a1 + 80);
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v29 = (_QWORD *)(a1 + 16);
      v30 = (_QWORD *)(a1 + 80);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (_QWORD *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 8;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = (_QWORD *)v6;
              v12 != j;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *j;
            v18 = *j;
            BYTE1(v18) = BYTE1(*j) & 0xEF;
            if ( v18 != -8192 )
              break;
            j += 2;
            if ( v12 == j )
              return (_QWORD *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v19 = a1 + 16;
            v20 = 3;
          }
          else
          {
            v28 = *(_DWORD *)(a1 + 24);
            v19 = *(_QWORD *)(a1 + 16);
            if ( !v28 )
              goto LABEL_74;
            v20 = v28 - 1;
          }
          v21 = 1;
          v22 = 0;
          v23 = v20 & (37 * v17);
          v24 = (__int64 *)(v19 + 16LL * v23);
          v25 = *v24;
          if ( v17 != *v24 )
          {
            while ( v25 != -4096 )
            {
              if ( !v22 && v25 == -8192 )
                v22 = v24;
              v23 = v20 & (v21 + v23);
              v24 = (__int64 *)(v19 + 16LL * v23);
              v25 = *v24;
              if ( v17 == *v24 )
                goto LABEL_19;
              ++v21;
            }
            if ( v22 )
              v24 = v22;
          }
LABEL_19:
          v26 = *j;
          j += 2;
          *v24 = v26;
          *((_DWORD *)v24 + 2) = *((_DWORD *)j - 2);
        }
        return (_QWORD *)sub_C7D6A0(v6, v10, 8);
      }
      v29 = (_QWORD *)(a1 + 16);
      v30 = (_QWORD *)(a1 + 80);
      v2 = 64;
    }
  }
  v31 = v29;
  v32 = v48;
  do
  {
    v33 = *v31;
    BYTE1(v33) = BYTE1(*v31) & 0xEF;
    if ( v33 != -8192 )
    {
      if ( v32 )
        *v32 = *v31;
      v32 += 2;
      *((_DWORD *)v32 - 2) = *((_DWORD *)v31 + 2);
    }
    v31 += 2;
  }
  while ( v31 != v30 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v34 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v34;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v35 = *(_QWORD **)(a1 + 16);
    v36 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v35 = v29;
    v36 = 8;
  }
  for ( k = &v35[v36]; k != v35; v35 += 2 )
  {
    if ( v35 )
      *v35 = -4096;
  }
  result = v48;
  if ( v32 != (_QWORD *)v48 )
  {
    do
    {
      while ( 1 )
      {
        v17 = *result;
        v38 = *result;
        BYTE1(v38) = BYTE1(*result) & 0xEF;
        if ( v38 != -8192 )
          break;
        result += 2;
        if ( v32 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v39 = v29;
        v40 = 3;
      }
      else
      {
        v47 = *(_DWORD *)(a1 + 24);
        v39 = *(_QWORD **)(a1 + 16);
        if ( !v47 )
        {
LABEL_74:
          MEMORY[0] = v17;
          BUG();
        }
        v40 = v47 - 1;
      }
      v41 = 1;
      v42 = 0;
      v43 = v40 & (37 * v17);
      v44 = &v39[2 * v43];
      v45 = *v44;
      if ( v17 != *v44 )
      {
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v42 )
            v42 = v44;
          v43 = v40 & (v41 + v43);
          v44 = &v39[2 * v43];
          v45 = *v44;
          if ( v17 == *v44 )
            goto LABEL_48;
          ++v41;
        }
        if ( v42 )
          v44 = v42;
      }
LABEL_48:
      v46 = *result;
      result += 2;
      *v44 = v46;
      *((_DWORD *)v44 + 2) = *((_DWORD *)result - 2);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v32 != result );
  }
  return result;
}
