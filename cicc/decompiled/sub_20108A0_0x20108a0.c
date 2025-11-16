// Function: sub_20108A0
// Address: 0x20108a0
//
__int64 __fastcall sub_20108A0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r15d
  __int64 v7; // r14
  _DWORD *v8; // rsi
  _DWORD *v9; // rax
  _DWORD *v10; // r12
  int v11; // edx
  _DWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _DWORD *v15; // rdx
  _DWORD *v16; // rdi
  int v17; // edx
  __int64 v18; // r9
  int v19; // edi
  int v20; // r11d
  _DWORD *v21; // r10
  unsigned int v22; // esi
  _DWORD *v23; // rcx
  int v24; // r8d
  int v25; // edx
  int *v26; // r12
  __int64 v27; // r13
  bool v28; // zf
  int *v29; // rcx
  _DWORD *v30; // rax
  __int64 v31; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  int v34; // edx
  __int64 v35; // r11
  int v36; // r8d
  int v37; // r14d
  _DWORD *v38; // r13
  unsigned int v39; // edi
  _DWORD *v40; // rsi
  int v41; // r9d
  int v42; // edx
  int v43; // esi
  __int64 v44; // rax
  int v45; // ecx
  _BYTE v46[112]; // [rsp+10h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v26 = *(int **)(a1 + 16);
    v27 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
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
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = 2LL * (unsigned int)v5;
      if ( v4 )
      {
LABEL_5:
        v8 = (_DWORD *)(a1 + 80);
        v9 = (_DWORD *)(a1 + 16);
        v10 = v46;
        do
        {
          while ( *v9 > 0xFFFFFFFD )
          {
            v9 += 2;
            if ( v9 == v8 )
              goto LABEL_11;
          }
          if ( v10 )
            *v10 = *v9;
          v11 = v9[1];
          v9 += 2;
          v10 += 2;
          *(v10 - 1) = v11;
        }
        while ( v9 != v8 );
LABEL_11:
        *(_BYTE *)(a1 + 8) &= ~1u;
        v12 = (_DWORD *)sub_22077B0(v7 * 4);
        v13 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v12;
        v14 = v13 & 1;
        *(_QWORD *)(a1 + 8) = v14;
        if ( (_BYTE)v14 )
        {
          v12 = (_DWORD *)(a1 + 16);
          v7 = 16;
        }
        v15 = v12;
        v16 = &v12[v7];
        while ( 1 )
        {
          if ( v15 )
            *v12 = -1;
          v12 += 2;
          if ( v16 == v12 )
            break;
          v15 = v12;
        }
        for ( result = (__int64)v46;
              v10 != (_DWORD *)result;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_DWORD *)result;
            if ( *(_DWORD *)result <= 0xFFFFFFFD )
              break;
            result += 8;
            if ( v10 == (_DWORD *)result )
              return result;
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 7;
          }
          else
          {
            v45 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v45 )
              goto LABEL_71;
            v19 = v45 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (_DWORD *)(v18 + 8LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (_DWORD *)(v18 + 8LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_25;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_25:
          *v23 = v17;
          v25 = *(_DWORD *)(result + 4);
          result += 8;
          v23[1] = v25;
        }
        return result;
      }
      v26 = *(int **)(a1 + 16);
      v27 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 128;
        v6 = 64;
        goto LABEL_5;
      }
      v26 = *(int **)(a1 + 16);
      v27 = *(unsigned int *)(a1 + 24);
      v7 = 128;
      v6 = 64;
    }
    v44 = sub_22077B0(v7 * 4);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v44;
  }
  v28 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v29 = &v26[2 * v27];
  if ( v28 )
  {
    v30 = *(_DWORD **)(a1 + 16);
    v31 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = (_DWORD *)(a1 + 16);
    v31 = 16;
  }
  for ( i = &v30[v31]; i != v30; v30 += 2 )
  {
    if ( v30 )
      *v30 = -1;
  }
  for ( j = v26; v29 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v34 = *j;
      if ( (unsigned int)*j <= 0xFFFFFFFD )
        break;
      j += 2;
      if ( v29 == j )
        return j___libc_free_0(v26);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = a1 + 16;
      v36 = 7;
    }
    else
    {
      v43 = *(_DWORD *)(a1 + 24);
      v35 = *(_QWORD *)(a1 + 16);
      if ( !v43 )
      {
LABEL_71:
        MEMORY[0] = 0;
        BUG();
      }
      v36 = v43 - 1;
    }
    v37 = 1;
    v38 = 0;
    v39 = v36 & (37 * v34);
    v40 = (_DWORD *)(v35 + 8LL * v39);
    v41 = *v40;
    if ( v34 != *v40 )
    {
      while ( v41 != -1 )
      {
        if ( v41 == -2 && !v38 )
          v38 = v40;
        v39 = v36 & (v37 + v39);
        v40 = (_DWORD *)(v35 + 8LL * v39);
        v41 = *v40;
        if ( v34 == *v40 )
          goto LABEL_42;
        ++v37;
      }
      if ( v38 )
        v40 = v38;
    }
LABEL_42:
    *v40 = v34;
    v42 = j[1];
    j += 2;
    v40[1] = v42;
  }
  return j___libc_free_0(v26);
}
