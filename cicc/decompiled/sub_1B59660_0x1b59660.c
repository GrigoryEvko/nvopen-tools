// Function: sub_1B59660
// Address: 0x1b59660
//
void __fastcall sub_1B59660(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // r15
  __int64 *v11; // r9
  __int64 *v12; // rbx
  _BYTE *v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rcx
  unsigned __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rcx
  _QWORD *v20; // rcx
  _QWORD *v21; // r8
  _BYTE *v22; // rbx
  _QWORD *v23; // r9
  int v24; // r8d
  int v25; // r11d
  __int64 *v26; // r10
  __int64 v27; // rcx
  __int64 *v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  int v33; // ecx
  unsigned int v34; // ebx
  bool v35; // zf
  __int64 v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v41; // r8
  int v42; // esi
  int v43; // r10d
  __int64 *v44; // r9
  __int64 v45; // rdx
  __int64 *v46; // rdi
  __int64 v47; // rcx
  unsigned __int64 v48; // rdi
  int v49; // edx
  __int64 v50; // rax
  __int64 *v51; // [rsp+10h] [rbp-1A0h]
  __int64 *v52; // [rsp+10h] [rbp-1A0h]
  _QWORD *v53; // [rsp+18h] [rbp-198h]
  _BYTE v54[400]; // [rsp+20h] [rbp-190h] BYREF

  v6 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( (_BYTE)v6 )
      return;
    v7 = *(_QWORD *)(a1 + 16);
    v34 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v9 = v8;
    if ( (unsigned int)v8 > 0x40 )
    {
      a5 = 11 * v8;
      v10 = 11LL * (unsigned int)v8;
      if ( (_BYTE)v6 )
        goto LABEL_5;
      v34 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( (_BYTE)v6 )
      {
        v10 = 704;
        v9 = 64;
LABEL_5:
        v11 = (__int64 *)(a1 + 368);
        v53 = (_QWORD *)(a1 + 16);
        v12 = (__int64 *)(a1 + 16);
        v13 = v54;
        do
        {
          v14 = *v12;
          if ( *v12 != -8 && v14 != -16 )
          {
            if ( v13 )
              *(_QWORD *)v13 = v14;
            *((_QWORD *)v13 + 1) = v13 + 24;
            v15 = *((unsigned int *)v12 + 4);
            *((_QWORD *)v13 + 2) = 0x400000000LL;
            if ( (_DWORD)v15 )
            {
              v52 = v11;
              sub_1B42B90((__int64)(v13 + 8), (__int64)(v12 + 1), v6, v15, a5, (int)v11);
              v11 = v52;
            }
            v16 = v12[1];
            v13 += 88;
            if ( (__int64 *)v16 != v12 + 3 )
            {
              v51 = v11;
              _libc_free(v16);
              v11 = v51;
            }
          }
          v12 += 11;
        }
        while ( v12 != v11 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v17 = (_QWORD *)sub_22077B0(v10 * 8);
        v18 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v9;
        *(_QWORD *)(a1 + 16) = v17;
        v19 = v18 & 1;
        if ( (_BYTE)v19 )
          v17 = v53;
        *(_QWORD *)(a1 + 8) = v19;
        if ( (_BYTE)v19 )
          v10 = 44;
        v20 = v17;
        v21 = &v17[v10];
        while ( 1 )
        {
          if ( v20 )
            *v17 = -8;
          v17 += 11;
          if ( v21 == v17 )
            break;
          v20 = v17;
        }
        v22 = v54;
        if ( v13 != v54 )
        {
          while ( 1 )
          {
            v32 = *(_QWORD *)v22;
            if ( *(_QWORD *)v22 != -8 && v32 != -16 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v23 = v53;
                v24 = 3;
              }
              else
              {
                v33 = *(_DWORD *)(a1 + 24);
                v23 = *(_QWORD **)(a1 + 16);
                if ( !v33 )
                  goto LABEL_85;
                v24 = v33 - 1;
              }
              v25 = 1;
              v26 = 0;
              v27 = v24 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
              v28 = &v23[11 * v27];
              v29 = *v28;
              if ( v32 != *v28 )
              {
                while ( v29 != -8 )
                {
                  if ( !v26 && v29 == -16 )
                    v26 = v28;
                  v27 = v24 & (unsigned int)(v25 + v27);
                  v28 = &v23[11 * (unsigned int)v27];
                  v29 = *v28;
                  if ( v32 == *v28 )
                    goto LABEL_28;
                  ++v25;
                }
                if ( v26 )
                  v28 = v26;
              }
LABEL_28:
              *v28 = v32;
              v28[1] = (__int64)(v28 + 3);
              v28[2] = 0x400000000LL;
              v30 = *((unsigned int *)v22 + 4);
              if ( (_DWORD)v30 )
                sub_1B42B90((__int64)(v28 + 1), (__int64)(v22 + 8), v30, v27, v24, (int)v23);
              v31 = *((_QWORD *)v22 + 1);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              if ( (_BYTE *)v31 != v22 + 24 )
                _libc_free(v31);
            }
            v22 += 88;
            if ( v13 == v22 )
              return;
          }
        }
        return;
      }
      v34 = *(_DWORD *)(a1 + 24);
      v10 = 704;
      v9 = 64;
    }
    v50 = sub_22077B0(v10 * 8);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v50;
  }
  v35 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v36 = v7 + 88LL * v34;
  if ( v35 )
  {
    v37 = *(_QWORD **)(a1 + 16);
    v38 = 11LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v37 = (_QWORD *)(a1 + 16);
    v38 = 44;
  }
  for ( i = &v37[v38]; i != v37; v37 += 11 )
  {
    if ( v37 )
      *v37 = -8;
  }
  for ( j = v7; v36 != j; j += 88 )
  {
    v32 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -16 && v32 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v41 = a1 + 16;
        v42 = 3;
      }
      else
      {
        v49 = *(_DWORD *)(a1 + 24);
        v41 = *(_QWORD *)(a1 + 16);
        if ( !v49 )
        {
LABEL_85:
          MEMORY[0] = v32;
          BUG();
        }
        v42 = v49 - 1;
      }
      v43 = 1;
      v44 = 0;
      v45 = v42 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v46 = (__int64 *)(v41 + 88 * v45);
      v47 = *v46;
      if ( *v46 != v32 )
      {
        while ( v47 != -8 )
        {
          if ( !v44 && v47 == -16 )
            v44 = v46;
          v45 = v42 & (unsigned int)(v43 + v45);
          v46 = (__int64 *)(v41 + 88LL * (unsigned int)v45);
          v47 = *v46;
          if ( v32 == *v46 )
            goto LABEL_51;
          ++v43;
        }
        if ( v44 )
          v46 = v44;
      }
LABEL_51:
      *v46 = v32;
      v46[1] = (__int64)(v46 + 3);
      v46[2] = 0x400000000LL;
      if ( *(_DWORD *)(j + 16) )
        sub_1B42B90((__int64)(v46 + 1), j + 8, v45, v47, v41, (int)v44);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v48 = *(_QWORD *)(j + 8);
      if ( v48 != j + 24 )
        _libc_free(v48);
    }
  }
  j___libc_free_0(v7);
}
