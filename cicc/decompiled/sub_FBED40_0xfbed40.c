// Function: sub_FBED40
// Address: 0xfbed40
//
void __fastcall sub_FBED40(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // rbx
  __int64 v7; // rcx
  unsigned int v8; // eax
  int v9; // eax
  __int64 v10; // rdx
  char **v11; // rax
  int v12; // eax
  char **v13; // r13
  char **v14; // r12
  char *v15; // r14
  char **v16; // r15
  char *v17; // rdx
  char **v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // r12
  char **v21; // r13
  __int64 v22; // rax
  char *v23; // rdi
  char *v24; // rax
  int v25; // edx
  int v26; // esi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  int v31; // [rsp+Ch] [rbp-74h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  char *v33[2]; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v34[88]; // [rsp+28h] [rbp-58h] BYREF

  v6 = (char **)a1;
  v7 = *(_DWORD *)(a1 + 8) & 0xFFFFFFFE;
  v8 = (_DWORD)a2[1] & 0xFFFFFFFE;
  *((_DWORD *)a2 + 2) = v7 | (_DWORD)a2[1] & 1;
  *(_DWORD *)(a1 + 8) = v8 | *(_DWORD *)(a1 + 8) & 1;
  v9 = *(_DWORD *)(a1 + 12);
  LODWORD(v10) = *((_DWORD *)a2 + 3);
  *(_DWORD *)(a1 + 12) = v10;
  *((_DWORD *)a2 + 3) = v9;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    if ( ((_BYTE)a2[1] & 1) == 0 )
      goto LABEL_4;
    v19 = a1 + 24;
    v20 = (__int64)(a2 + 3);
    v21 = a2 + 31;
    while ( 1 )
    {
      v22 = *(_QWORD *)(v19 - 8);
      LOBYTE(v10) = v22 != -4096;
      LOBYTE(v7) = v22 != -8192;
      v10 = (unsigned int)v7 & (unsigned int)v10;
      v7 = *(_QWORD *)(v20 - 8);
      if ( v7 == -4096 )
      {
        *(_QWORD *)(v19 - 8) = -4096;
        *(_QWORD *)(v20 - 8) = v22;
        if ( !(_BYTE)v10 )
          goto LABEL_22;
      }
      else
      {
        if ( v7 != -8192 )
        {
          if ( (_BYTE)v10 )
          {
            v26 = *(_DWORD *)(v19 + 8);
            v32 = *(_QWORD *)(v19 - 8);
            v33[0] = v34;
            v33[1] = (char *)0x400000000LL;
            if ( v26 )
            {
              sub_F8EFD0((__int64)v33, (char **)v19, v10, v7, a5, a6);
              v7 = *(_QWORD *)(v20 - 8);
            }
            *(_QWORD *)(v19 - 8) = v7;
            sub_F8EFD0(v19, (char **)v20, v10, v7, a5, a6);
            a2 = v33;
            *(_QWORD *)(v20 - 8) = v32;
            sub_F8EFD0(v20, v33, v27, v28, v29, v30);
            v23 = v33[0];
            if ( v33[0] == v34 )
              goto LABEL_22;
          }
          else
          {
            *(_QWORD *)(v19 - 8) = v7;
            *(_QWORD *)(v20 - 8) = v22;
            *(_QWORD *)v19 = v19 + 16;
            *(_DWORD *)(v19 + 8) = 0;
            *(_DWORD *)(v19 + 12) = 4;
            if ( *(_DWORD *)(v20 + 8) )
            {
              a2 = (char **)v20;
              sub_F8EFD0(v19, (char **)v20, v10, v7, a5, a6);
            }
            v23 = *(char **)v20;
            if ( *(_QWORD *)v20 == v20 + 16 )
              goto LABEL_22;
          }
          goto LABEL_21;
        }
        *(_QWORD *)(v19 - 8) = -8192;
        *(_QWORD *)(v20 - 8) = v22;
        if ( !(_BYTE)v10 )
          goto LABEL_22;
      }
      *(_DWORD *)(v20 + 8) = 0;
      *(_QWORD *)v20 = v20 + 16;
      *(_DWORD *)(v20 + 12) = 4;
      v10 = *(unsigned int *)(v19 + 8);
      if ( (_DWORD)v10 )
      {
        a2 = (char **)v19;
        sub_F8EFD0(v20, (char **)v19, v10, v7, a5, a6);
      }
      v23 = *(char **)v19;
      if ( *(_QWORD *)v19 == v19 + 16 )
        goto LABEL_22;
LABEL_21:
      _libc_free(v23, a2);
LABEL_22:
      v20 += 56;
      v19 += 56;
      if ( v21 == (char **)v20 )
        return;
    }
  }
  if ( ((_BYTE)a2[1] & 1) == 0 )
  {
    v24 = *(char **)(a1 + 16);
    *(_QWORD *)(a1 + 16) = a2[2];
    v25 = *((_DWORD *)a2 + 6);
    a2[2] = v24;
    LODWORD(v24) = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 24) = v25;
    *((_DWORD *)a2 + 6) = (_DWORD)v24;
    return;
  }
  v11 = a2;
  a2 = (char **)a1;
  v6 = v11;
LABEL_4:
  v12 = *((_DWORD *)a2 + 6);
  *((_BYTE *)a2 + 8) |= 1u;
  v13 = v6 + 2;
  v14 = a2 + 2;
  v15 = a2[2];
  v16 = a2 + 30;
  v31 = v12;
  do
  {
    v17 = *v13;
    *v14 = *v13;
    if ( v17 != (char *)-8192LL && v17 != (char *)-4096LL )
    {
      *((_DWORD *)v14 + 4) = 0;
      v14[1] = (char *)(v14 + 3);
      *((_DWORD *)v14 + 5) = 4;
      if ( *((_DWORD *)v13 + 4) )
      {
        a2 = v13 + 1;
        sub_F8EFD0((__int64)(v14 + 1), v13 + 1, (__int64)(v14 + 3), v7, a5, a6);
      }
      v18 = (char **)v13[1];
      if ( v18 != v13 + 3 )
        _libc_free(v18, a2);
    }
    v14 += 7;
    v13 += 7;
  }
  while ( v14 != v16 );
  *((_BYTE *)v6 + 8) &= ~1u;
  v6[2] = v15;
  *((_DWORD *)v6 + 6) = v31;
}
