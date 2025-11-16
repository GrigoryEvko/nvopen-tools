// Function: sub_141AF30
// Address: 0x141af30
//
__int64 __fastcall sub_141AF30(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // ebx
  __int64 v8; // r13
  __int64 v9; // r12
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rax
  unsigned int v15; // r15d
  bool v16; // zf
  __int64 *v17; // rbx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  __int64 *j; // r13
  __int64 v22; // rax
  __int64 v23; // rdi
  int v24; // esi
  int v25; // r10d
  _QWORD *v26; // r9
  unsigned int v27; // ecx
  _QWORD *v28; // rdx
  __int64 v29; // r8
  __int64 v30; // rdi
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-198h]
  _BYTE v36[400]; // [rsp+10h] [rbp-190h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v15 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = *(__int64 **)(a1 + 16);
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v7 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v8 = 88LL * (unsigned int)v6;
      if ( v4 )
      {
LABEL_5:
        v9 = a1 + 16;
        v35 = a1 + 368;
        v10 = v36;
        do
        {
          v11 = *(_QWORD *)v9;
          if ( *(_QWORD *)v9 != -8 && v11 != -16 )
          {
            if ( v10 )
              *v10 = v11;
            v10[1] = 0;
            v12 = (__int64)(v10 + 1);
            v13 = v10 + 3;
            v10 += 11;
            *(v10 - 9) = 1;
            do
            {
              if ( v13 )
                *v13 = -8;
              v13 += 2;
            }
            while ( v13 != v10 );
            sub_1415470(v12, v9 + 8);
            if ( (*(_BYTE *)(v9 + 16) & 1) == 0 )
              j___libc_free_0(*(_QWORD *)(v9 + 24));
          }
          v9 += 88;
        }
        while ( v9 != v35 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v14 = sub_22077B0(v8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v14;
        return sub_141AD70(a1, (__int64)v36, (__int64)v10);
      }
      v15 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 5632;
        v7 = 64;
        goto LABEL_5;
      }
      v15 = *(_DWORD *)(a1 + 24);
      v8 = 5632;
      v7 = 64;
    }
    v34 = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v34;
  }
  v16 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v17 = &v5[11 * v15];
  if ( v16 )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = 11LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v18 = (_QWORD *)(a1 + 16);
    v19 = 44;
  }
  for ( i = &v18[v19]; i != v18; v18 += 11 )
  {
    if ( v18 )
      *v18 = -8;
  }
  for ( j = v5; v17 != j; j += 11 )
  {
    v22 = *j;
    if ( *j != -8 && v22 != -16 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v23 = a1 + 16;
        v24 = 3;
      }
      else
      {
        v33 = *(_DWORD *)(a1 + 24);
        v23 = *(_QWORD *)(a1 + 16);
        if ( !v33 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v24 = v33 - 1;
      }
      v25 = 1;
      v26 = 0;
      v27 = v24 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v28 = (_QWORD *)(v23 + 88LL * v27);
      v29 = *v28;
      if ( *v28 != v22 )
      {
        while ( v29 != -8 )
        {
          if ( !v26 && v29 == -16 )
            v26 = v28;
          v27 = v24 & (v25 + v27);
          v28 = (_QWORD *)(v23 + 88LL * v27);
          v29 = *v28;
          if ( v22 == *v28 )
            goto LABEL_33;
          ++v25;
        }
        if ( v26 )
          v28 = v26;
      }
LABEL_33:
      *v28 = v22;
      v30 = (__int64)(v28 + 1);
      v31 = v28 + 3;
      v32 = v28 + 11;
      *(v32 - 10) = 0;
      *(v32 - 9) = 1;
      do
      {
        if ( v31 )
          *v31 = -8;
        v31 += 2;
      }
      while ( v31 != v32 );
      sub_1415470(v30, (__int64)(j + 1));
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( (j[2] & 1) == 0 )
        j___libc_free_0(j[3]);
    }
  }
  return j___libc_free_0(v5);
}
