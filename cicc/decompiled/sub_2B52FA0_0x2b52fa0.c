// Function: sub_2B52FA0
// Address: 0x2b52fa0
//
void __fastcall sub_2B52FA0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  __int64 v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  char **v29; // rbx
  char **v30; // r15
  char **v31; // r13
  char *v32; // rsi
  __int64 v33; // rcx
  unsigned __int64 v34; // rdi
  char *v35; // rax
  __int64 v36; // rax
  char **j; // rax
  char **v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  char **v42; // rax
  _QWORD *v43; // rdi
  _QWORD *v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rdi
  char *v47; // rax
  __int64 v48; // rax
  __int64 v49; // [rsp+8h] [rbp-308h]
  __int64 v50; // [rsp+8h] [rbp-308h]
  _QWORD *v51; // [rsp+18h] [rbp-2F8h] BYREF
  _QWORD v52[94]; // [rsp+20h] [rbp-2F0h] BYREF

  v7 = a2;
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !(_BYTE)v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
LABEL_8:
      v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v49 = 88LL * v11;
      v15 = v10 + v49;
      if ( v14 )
      {
        v16 = *(_QWORD **)(a1 + 16);
        v17 = 11LL * *(unsigned int *)(a1 + 24);
      }
      else
      {
        v16 = (_QWORD *)(a1 + 16);
        v17 = 88;
      }
      for ( i = &v16[v17]; i != v16; v16 += 11 )
      {
        if ( v16 )
        {
          *v16 = -4096;
          v16[1] = -4096;
          v16[2] = -4096;
        }
      }
      v19 = v10;
      if ( v15 == v10 )
      {
LABEL_46:
        sub_C7D6A0(v10, v49, 8);
        return;
      }
      while ( 1 )
      {
        v28 = *(_QWORD *)(v19 + 16);
        if ( v28 == -4096 )
        {
          if ( *(_QWORD *)(v19 + 8) != -4096 || *(_QWORD *)v19 != -4096 )
            goto LABEL_17;
        }
        else if ( v28 != -8192 || *(_QWORD *)(v19 + 8) != -8192 || *(_QWORD *)v19 != -8192 )
        {
LABEL_17:
          sub_2B3EEF0(a1, (__int64 *)v19, v52);
          v23 = (_QWORD *)v52[0];
          *(_QWORD *)(v52[0] + 16LL) = *(_QWORD *)(v19 + 16);
          v23[1] = *(_QWORD *)(v19 + 8);
          v24 = *(_QWORD *)v19;
          *v23 = *(_QWORD *)v19;
          v25 = v52[0];
          v26 = v52[0] + 40LL;
          *(_QWORD *)(v52[0] + 32LL) = 0x600000000LL;
          *(_QWORD *)(v25 + 24) = v26;
          if ( *(_DWORD *)(v19 + 32) )
            sub_2B0B710(v25 + 24, (char **)(v19 + 24), v24, v20, v21, v22);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          v27 = *(_QWORD *)(v19 + 24);
          if ( v27 != v19 + 40 )
            _libc_free(v27);
        }
        v19 += 88;
        if ( v15 == v19 )
          goto LABEL_46;
      }
    }
  }
  else
  {
    v9 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v7 = v9;
    if ( (unsigned int)v9 <= 0x40 )
    {
      if ( !(_BYTE)v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(_DWORD *)(a1 + 24);
        v7 = 64;
        v12 = 5632;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v13;
        goto LABEL_8;
      }
      v29 = (char **)(a1 + 16);
      v7 = 64;
      v50 = a1 + 720;
      goto LABEL_27;
    }
    if ( !(_BYTE)v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 88LL * (unsigned int)v9;
      goto LABEL_5;
    }
  }
  v29 = (char **)(a1 + 16);
  v50 = a1 + 720;
LABEL_27:
  v30 = v29;
  v31 = (char **)v52;
  do
  {
    v35 = v30[2];
    if ( v35 == (char *)-4096LL )
    {
      if ( v30[1] == (char *)-4096LL && *v30 == (char *)-4096LL )
        goto LABEL_35;
    }
    else if ( v35 == (char *)-8192LL && v30[1] == (char *)-8192LL && *v30 == (char *)-8192LL )
    {
      goto LABEL_35;
    }
    if ( v31 )
    {
      v32 = *v30;
      v31[2] = v35;
      *v31 = v32;
      v31[1] = v30[1];
    }
    v33 = *((unsigned int *)v30 + 8);
    v31[3] = (char *)(v31 + 5);
    v31[4] = (char *)0x600000000LL;
    if ( (_DWORD)v33 )
      sub_2B0B710((__int64)(v31 + 3), v30 + 3, v8, v33, a5, a6);
    v34 = (unsigned __int64)v30[3];
    v31 += 11;
    if ( (char **)v34 != v30 + 5 )
      _libc_free(v34);
LABEL_35:
    v30 += 11;
  }
  while ( v30 != (char **)v50 );
  if ( v7 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v48 = sub_C7D670(88LL * v7, 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v48;
  }
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v36 = 88;
  if ( v14 )
  {
    v29 = *(char ***)(a1 + 16);
    v36 = 11LL * *(unsigned int *)(a1 + 24);
  }
  for ( j = &v29[v36]; j != v29; v29 += 11 )
  {
    if ( v29 )
    {
      *v29 = (char *)-4096LL;
      v29[1] = (char *)-4096LL;
      v29[2] = (char *)-4096LL;
    }
  }
  v38 = (char **)v52;
  if ( v31 != v52 )
  {
    do
    {
      v47 = v38[2];
      if ( v47 == (char *)-4096LL )
      {
        if ( v38[1] != (char *)-4096LL || *v38 != (char *)-4096LL )
          goto LABEL_59;
      }
      else if ( v47 != (char *)-8192LL || v38[1] != (char *)-8192LL || *v38 != (char *)-8192LL )
      {
LABEL_59:
        sub_2B3EEF0(a1, (__int64 *)v38, &v51);
        v42 = (char **)v51;
        v51[2] = v38[2];
        v42[1] = v38[1];
        *v42 = *v38;
        v43 = v51;
        v44 = v51 + 5;
        v51[4] = 0x600000000LL;
        v43[3] = v44;
        v45 = *((unsigned int *)v38 + 8);
        if ( (_DWORD)v45 )
          sub_2B0B710((__int64)(v43 + 3), v38 + 3, v45, v39, v40, v41);
        v46 = (unsigned __int64)v38[3];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        if ( (char **)v46 != v38 + 5 )
          _libc_free(v46);
      }
      v38 += 11;
    }
    while ( v31 != v38 );
  }
}
