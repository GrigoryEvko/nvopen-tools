// Function: sub_2BA2A80
// Address: 0x2ba2a80
//
__int64 __fastcall sub_2BA2A80(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r13
  bool v12; // zf
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  __int64 v17; // rbx
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 result; // rax
  __int64 *v21; // r12
  __int64 v22; // rsi
  __int64 *v23; // rbx
  __int64 *v24; // rax
  __int64 *v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-148h]
  __int64 v33; // [rsp+18h] [rbp-138h] BYREF
  _QWORD v34[38]; // [rsp+20h] [rbp-130h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      v32 = v10;
LABEL_8:
      v11 = 32 * v7;
      v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v13 = v32 + v11;
      if ( v12 )
      {
        v14 = *(_QWORD **)(a1 + 16);
        v15 = 4LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        v14 = (_QWORD *)(a1 + 16);
        v15 = 32;
      }
      for ( i = &v14[v15]; i != v14; v14 += 4 )
      {
        if ( v14 )
        {
          *v14 = -4096;
          v14[1] = -4096;
          v14[2] = -4096;
        }
      }
      v17 = v32;
      if ( v13 == v32 )
        return sub_C7D6A0(v32, v11, 8);
      while ( 1 )
      {
        v19 = *(_QWORD *)(v17 + 16);
        if ( v19 != -4096 )
          break;
        if ( *(_QWORD *)(v17 + 8) == -4096 && *(_QWORD *)v17 == -4096 )
        {
          v17 += 32;
          if ( v13 == v17 )
            return sub_C7D6A0(v32, v11, 8);
        }
        else
        {
LABEL_17:
          sub_2B47D60(a1, (__int64 *)v17, v34);
          v18 = (_QWORD *)v34[0];
          *(_QWORD *)(v34[0] + 16LL) = *(_QWORD *)(v17 + 16);
          v18[1] = *(_QWORD *)(v17 + 8);
          *v18 = *(_QWORD *)v17;
          *(_DWORD *)(v34[0] + 24LL) = *(_DWORD *)(v17 + 24);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_18:
          v17 += 32;
          if ( v13 == v17 )
            return sub_C7D6A0(v32, v11, 8);
        }
      }
      if ( v19 == -8192 && *(_QWORD *)(v17 + 8) == -8192 && *(_QWORD *)v17 == -8192 )
        goto LABEL_18;
      goto LABEL_17;
    }
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
    if ( (unsigned int)v5 <= 0x40 )
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(unsigned int *)(a1 + 24);
        v2 = 64;
        v8 = 2048;
        v32 = v6;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
        goto LABEL_8;
      }
      v21 = (__int64 *)(a1 + 16);
      v22 = a1 + 272;
      v2 = 64;
      goto LABEL_25;
    }
    if ( !v4 )
    {
      v30 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v8 = 32LL * (unsigned int)v5;
      v32 = v30;
      goto LABEL_5;
    }
  }
  v21 = (__int64 *)(a1 + 16);
  v22 = a1 + 272;
LABEL_25:
  v23 = v34;
  v24 = v21;
  v25 = v34;
  do
  {
    v27 = v24[2];
    if ( v27 == -4096 )
    {
      if ( v24[1] == -4096 && *v24 == -4096 )
        goto LABEL_30;
    }
    else if ( v27 == -8192 && v24[1] == -8192 && *v24 == -8192 )
    {
      goto LABEL_30;
    }
    if ( v25 )
    {
      v26 = *v24;
      v25[2] = v27;
      *v25 = v26;
      v25[1] = v24[1];
    }
    v25 += 4;
    *((_DWORD *)v25 - 2) = *((_DWORD *)v24 + 6);
LABEL_30:
    v24 += 4;
  }
  while ( v24 != (__int64 *)v22 );
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v31 = sub_C7D670(32LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v31;
  }
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v28 = 32;
  if ( v12 )
  {
    v21 = *(__int64 **)(a1 + 16);
    v28 = 4LL * *(unsigned int *)(a1 + 24);
  }
  for ( result = (__int64)&v21[v28]; (__int64 *)result != v21; v21 += 4 )
  {
    if ( v21 )
    {
      *v21 = -4096;
      v21[1] = -4096;
      v21[2] = -4096;
    }
  }
  if ( v25 != v34 )
  {
    do
    {
      result = v23[2];
      if ( result == -4096 )
      {
        if ( v23[1] == -4096 && *v23 == -4096 )
          goto LABEL_54;
      }
      else if ( result == -8192 && v23[1] == -8192 && *v23 == -8192 )
      {
        goto LABEL_54;
      }
      sub_2B47D60(a1, v23, &v33);
      v29 = (__int64 *)v33;
      *(_QWORD *)(v33 + 16) = v23[2];
      v29[1] = v23[1];
      *v29 = *v23;
      *(_DWORD *)(v33 + 24) = *((_DWORD *)v23 + 6);
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
LABEL_54:
      v23 += 4;
    }
    while ( v25 != v23 );
  }
  return result;
}
