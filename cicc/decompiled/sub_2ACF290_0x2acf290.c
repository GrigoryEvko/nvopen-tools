// Function: sub_2ACF290
// Address: 0x2acf290
//
void __fastcall sub_2ACF290(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  char v8; // bl
  unsigned int v9; // eax
  __int64 v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  bool v15; // zf
  __int64 v16; // r15
  _DWORD *v17; // rax
  __int64 v18; // rdx
  _DWORD *i; // rdx
  __int64 v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdx
  int v24; // r11d
  unsigned int *v25; // r10
  __int64 v26; // rcx
  unsigned int *v27; // rdi
  __int64 v28; // r9
  unsigned __int64 v29; // rdi
  int v30; // edx
  char **v31; // r15
  char **v32; // rbx
  char **v33; // r14
  __int64 v34; // rcx
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  _DWORD *v37; // rax
  __int64 v38; // rcx
  _DWORD *j; // rcx
  char **v40; // rbx
  int v41; // eax
  __int64 v42; // r9
  __int64 v43; // r8
  int v44; // r11d
  __int64 v45; // r10
  __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // esi
  __int64 v49; // rdx
  unsigned __int64 v50; // rdi
  int v51; // ecx
  char **v52; // [rsp+8h] [rbp-2B8h]
  _BYTE v53[688]; // [rsp+10h] [rbp-2B0h] BYREF

  v6 = a2;
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v31 = (char **)(a1 + 656);
    v52 = (char **)(a1 + 16);
  }
  else
  {
    v9 = sub_AF1560(a2 - 1);
    v6 = v9;
    if ( v9 > 0x40 )
    {
      v31 = (char **)(a1 + 656);
      v52 = (char **)(a1 + 16);
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 40LL * v9;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 2560;
        v6 = 64;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
        v14 = 40LL * v11;
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v10 + v14;
        if ( v15 )
        {
          v17 = *(_DWORD **)(a1 + 16);
          v18 = 10LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v17 = (_DWORD *)(a1 + 16);
          v18 = 160;
        }
        for ( i = &v17[v18]; i != v17; v17 += 10 )
        {
          if ( v17 )
            *v17 = -1;
        }
        v20 = v10;
        if ( v16 != v10 )
        {
          while ( 1 )
          {
            v21 = *(_DWORD *)v20;
            if ( *(_DWORD *)v20 > 0xFFFFFFFD )
              goto LABEL_16;
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v22 = a1 + 16;
              v23 = 15;
            }
            else
            {
              v30 = *(_DWORD *)(a1 + 24);
              v22 = *(_QWORD *)(a1 + 16);
              if ( !v30 )
                goto LABEL_83;
              v23 = (unsigned int)(v30 - 1);
            }
            v24 = 1;
            v25 = 0;
            v26 = (unsigned int)v23 & (37 * v21);
            v27 = (unsigned int *)(v22 + 40 * v26);
            v28 = *v27;
            if ( v21 != (_DWORD)v28 )
            {
              while ( (_DWORD)v28 != -1 )
              {
                if ( (_DWORD)v28 == -2 && !v25 )
                  v25 = v27;
                a5 = (unsigned int)(v24 + 1);
                v26 = (unsigned int)v23 & (v24 + (_DWORD)v26);
                v27 = (unsigned int *)(v22 + 40LL * (unsigned int)v26);
                v28 = *v27;
                if ( v21 == (_DWORD)v28 )
                  goto LABEL_21;
                ++v24;
              }
              if ( v25 )
                v27 = v25;
            }
LABEL_21:
            *v27 = v21;
            *((_QWORD *)v27 + 1) = v27 + 6;
            *((_QWORD *)v27 + 2) = 0x200000000LL;
            if ( *(_DWORD *)(v20 + 16) )
              sub_2AA8B90((__int64)(v27 + 2), (char **)(v20 + 8), v23, v26, a5, v28);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v29 = *(_QWORD *)(v20 + 8);
            if ( v29 == v20 + 24 )
            {
LABEL_16:
              v20 += 40;
              if ( v16 == v20 )
                break;
            }
            else
            {
              _libc_free(v29);
              v20 += 40;
              if ( v16 == v20 )
                break;
            }
          }
        }
        sub_C7D6A0(v10, v14, 8);
        return;
      }
      v31 = (char **)(a1 + 656);
      v6 = 64;
      v52 = (char **)(a1 + 16);
    }
  }
  v32 = v52;
  v33 = (char **)v53;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)v32 <= 0xFFFFFFFD )
      {
        if ( v33 )
          *(_DWORD *)v33 = *(_DWORD *)v32;
        v33[1] = (char *)(v33 + 3);
        v34 = *((unsigned int *)v32 + 4);
        v33[2] = (char *)0x200000000LL;
        if ( (_DWORD)v34 )
          sub_2AA8B90((__int64)(v33 + 1), v32 + 1, a3, v34, a5, a6);
        v35 = (unsigned __int64)v32[1];
        v33 += 5;
        if ( (char **)v35 != v32 + 3 )
          break;
      }
      v32 += 5;
      if ( v32 == v31 )
        goto LABEL_39;
    }
    _libc_free(v35);
    v32 += 5;
  }
  while ( v32 != v31 );
LABEL_39:
  if ( v6 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v36 = sub_C7D670(40LL * v6, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v36;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    v37 = *(_DWORD **)(a1 + 16);
    v38 = 10LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v37 = v52;
    v38 = 160;
  }
  for ( j = &v37[v38]; j != v37; v37 += 10 )
  {
    if ( v37 )
      *v37 = -1;
  }
  v40 = (char **)v53;
  if ( v33 != (char **)v53 )
  {
    do
    {
      while ( 1 )
      {
        v41 = *(_DWORD *)v40;
        if ( *(_DWORD *)v40 <= 0xFFFFFFFD )
        {
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v42 = (__int64)v52;
            v43 = 15;
          }
          else
          {
            v51 = *(_DWORD *)(a1 + 24);
            v42 = *(_QWORD *)(a1 + 16);
            if ( !v51 )
            {
LABEL_83:
              MEMORY[0] = 0;
              BUG();
            }
            v43 = (unsigned int)(v51 - 1);
          }
          v44 = 1;
          v45 = 0;
          v46 = (unsigned int)v43 & (37 * v41);
          v47 = v42 + 40 * v46;
          v48 = *(_DWORD *)v47;
          if ( v41 != *(_DWORD *)v47 )
          {
            while ( v48 != -1 )
            {
              if ( v48 == -2 && !v45 )
                v45 = v47;
              v46 = (unsigned int)v43 & (v44 + (_DWORD)v46);
              v47 = v42 + 40LL * (unsigned int)v46;
              v48 = *(_DWORD *)v47;
              if ( v41 == *(_DWORD *)v47 )
                goto LABEL_54;
              ++v44;
            }
            if ( v45 )
              v47 = v45;
          }
LABEL_54:
          *(_DWORD *)v47 = v41;
          *(_QWORD *)(v47 + 8) = v47 + 24;
          *(_QWORD *)(v47 + 16) = 0x200000000LL;
          v49 = *((unsigned int *)v40 + 4);
          if ( (_DWORD)v49 )
            sub_2AA8B90(v47 + 8, v40 + 1, v49, v46, v43, v42);
          v50 = (unsigned __int64)v40[1];
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          if ( (char **)v50 != v40 + 3 )
            break;
        }
        v40 += 5;
        if ( v33 == v40 )
          return;
      }
      _libc_free(v50);
      v40 += 5;
    }
    while ( v33 != v40 );
  }
}
