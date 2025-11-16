// Function: sub_2862850
// Address: 0x2862850
//
void __fastcall sub_2862850(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rbx
  int v24; // esi
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+18h] [rbp-A8h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28[3]; // [rsp+28h] [rbp-98h] BYREF
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v30; // [rsp+60h] [rbp-60h] BYREF
  __int64 v31; // [rsp+68h] [rbp-58h]
  _QWORD v32[10]; // [rsp+70h] [rbp-50h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( v5 )
  {
    v10 = *(unsigned int *)(a1 + 24);
    v11 = 48 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v30 = v32;
    v12 = v5 + 48 * v4;
    v31 = 0x400000001LL;
    v13 = v7 + 48 * v10;
    v32[0] = -1;
    if ( v7 != v13 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            v14 = (unsigned int)v31;
            *(_DWORD *)(v7 + 12) = 4;
            if ( (_DWORD)v14 )
              break;
          }
          v7 += 48;
          if ( v13 == v7 )
            goto LABEL_10;
        }
        v25 = v13;
        v26 = v7;
        sub_2850210(v7, (__int64)&v30, v13, v14, v8, v9);
        v13 = v25;
        v7 = v26 + 48;
      }
      while ( v25 != v26 + 48 );
LABEL_10:
      if ( v30 != v32 )
        _libc_free((unsigned __int64)v30);
    }
    v30 = v32;
    v15 = v5;
    v28[1] = (__int64)&v29;
    v28[2] = 0x400000001LL;
    v31 = 0x400000001LL;
    v29 = -1;
    for ( v32[0] = -2; v12 != v15; v15 += 48 )
    {
      if ( *(_DWORD *)(v15 + 8) != 1 || (v20 = *(char **)v15, **(_QWORD **)v15 != v29) && v32[0] != *(_QWORD *)v20 )
      {
        sub_28626B0(a1, v15, v28);
        sub_28502F0(v28[0], (char **)v15, v16, v17, v18, v19);
        ++*(_DWORD *)(a1 + 16);
        v20 = *(char **)v15;
      }
      if ( v20 != (char *)(v15 + 16) )
        _libc_free((unsigned __int64)v20);
    }
    sub_C7D6A0(v5, v11, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v22 = 0x400000001LL;
    v30 = v32;
    v31 = 0x400000001LL;
    v32[0] = -1;
    v23 = v7 + 48 * v21;
    if ( v7 != v23 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            v24 = v31;
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            *(_DWORD *)(v7 + 12) = 4;
            if ( v24 )
              break;
          }
          v7 += 48;
          if ( v23 == v7 )
            goto LABEL_29;
        }
        v27 = v7;
        sub_2850210(v7, (__int64)&v30, v7 + 16, v22, v8, v9);
        v7 = v27 + 48;
      }
      while ( v23 != v27 + 48 );
LABEL_29:
      if ( v30 != v32 )
        _libc_free((unsigned __int64)v30);
    }
  }
}
