// Function: sub_2865FC0
// Address: 0x2865fc0
//
void __fastcall sub_2865FC0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+18h] [rbp-A8h]
  __int64 v26[3]; // [rsp+28h] [rbp-98h] BYREF
  __int64 v27; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v28; // [rsp+60h] [rbp-60h] BYREF
  __int64 v29; // [rsp+68h] [rbp-58h]
  _QWORD v30[10]; // [rsp+70h] [rbp-50h] BYREF

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
  v7 = sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = v30;
    v30[0] = -1;
    v23 = 56 * v4;
    v10 = v5 + 56 * v4;
    v29 = 0x400000001LL;
    v11 = *(unsigned int *)(a1 + 24);
    v12 = v7 + 56 * v11;
    if ( v7 != v12 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            v13 = (unsigned int)v29;
            *(_DWORD *)(v7 + 12) = 4;
            if ( (_DWORD)v13 )
              break;
          }
          v7 += 56;
          if ( v12 == v7 )
            goto LABEL_10;
        }
        v24 = v7;
        sub_2850210(v7, (__int64)&v28, v13, v11, v8, v9);
        v7 = v24 + 56;
      }
      while ( v12 != v24 + 56 );
LABEL_10:
      if ( v28 != v30 )
        _libc_free((unsigned __int64)v28);
    }
    v28 = v30;
    v14 = v5;
    v26[1] = (__int64)&v27;
    v26[2] = 0x400000001LL;
    v27 = -1;
    v29 = 0x400000001LL;
    for ( v30[0] = -2; v10 != v14; v14 += 56 )
    {
      if ( *(_DWORD *)(v14 + 8) != 1 || (v19 = *(char **)v14, **(_QWORD **)v14 != v27) && v30[0] != *(_QWORD *)v19 )
      {
        sub_2865E20(a1, v14, v26);
        sub_28502F0(v26[0], (char **)v14, v15, v16, v17, v18);
        *(_QWORD *)(v26[0] + 48) = *(_QWORD *)(v14 + 48);
        ++*(_DWORD *)(a1 + 16);
        v19 = *(char **)v14;
      }
      if ( v19 != (char *)(v14 + 16) )
        _libc_free((unsigned __int64)v19);
    }
    sub_C7D6A0(v5, v23, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 0x400000001LL;
    v20 = *(unsigned int *)(a1 + 24);
    v28 = v30;
    v30[0] = -1;
    v21 = v7 + 56 * v20;
    if ( v7 != v21 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            v22 = (unsigned int)v29;
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            *(_DWORD *)(v7 + 12) = 4;
            if ( (_DWORD)v22 )
              break;
          }
          v7 += 56;
          if ( v21 == v7 )
            goto LABEL_29;
        }
        v25 = v7;
        sub_2850210(v7, (__int64)&v28, v7 + 16, v22, v8, v9);
        v7 = v25 + 56;
      }
      while ( v21 != v25 + 56 );
LABEL_29:
      if ( v28 != v30 )
        _libc_free((unsigned __int64)v28);
    }
  }
}
