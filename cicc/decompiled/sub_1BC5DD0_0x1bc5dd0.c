// Function: sub_1BC5DD0
// Address: 0x1bc5dd0
//
void __fastcall sub_1BC5DD0(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rbx
  __int64 v11; // r14
  _DWORD **v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  char *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-90h]
  _DWORD **v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30[3]; // [rsp+18h] [rbp-78h] BYREF
  int v31; // [rsp+30h] [rbp-60h] BYREF
  _DWORD *v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  _DWORD v34[16]; // [rsp+50h] [rbp-40h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  v10 = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v11 = v4 + 40 * v3;
    v32 = v34;
    v12 = &v32;
    v33 = 0x400000001LL;
    v13 = *(unsigned int *)(a1 + 24);
    v34[0] = -2;
    v14 = v10 + 40 * v13;
    if ( v10 != v14 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            *(_DWORD *)(v10 + 8) = 0;
            *(_QWORD *)v10 = v10 + 16;
            v15 = v33;
            *(_DWORD *)(v10 + 12) = 4;
            if ( v15 )
              break;
          }
          v10 += 40;
          if ( v14 == v10 )
            goto LABEL_10;
        }
        v16 = v10;
        v27 = v14;
        v10 += 40;
        v28 = v12;
        sub_1BB9EE0(v16, (__int64)v12, v14, (__int64)v12, v8, v9);
        v14 = v27;
        v12 = v28;
      }
      while ( v27 != v10 );
LABEL_10:
      if ( v32 != v34 )
        _libc_free((unsigned __int64)v32);
    }
    v32 = v34;
    v17 = v4;
    v30[1] = (__int64)&v31;
    v30[2] = 0x400000001LL;
    v31 = -2;
    v33 = 0x400000001LL;
    for ( v34[0] = -3; v11 != v17; v17 += 40 )
    {
      if ( *(_DWORD *)(v17 + 8) != 1 || (v22 = *(char **)v17, **(_DWORD **)v17 != v31) && v34[0] != *(_DWORD *)v22 )
      {
        sub_1BC2430(a1, v17, v30);
        v29 = v30[0];
        sub_1BB9DA0(v30[0], (char **)v17, v18, v19, v20, v21);
        *(_DWORD *)(v29 + 32) = *(_DWORD *)(v17 + 32);
        ++*(_DWORD *)(a1 + 16);
        v22 = *(char **)v17;
      }
      if ( v22 != (char *)(v17 + 16) )
        _libc_free((unsigned __int64)v22);
    }
    j___libc_free_0(v4);
  }
  else
  {
    v34[0] = -2;
    v33 = 0x400000001LL;
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v32 = v34;
    v24 = v10 + 40 * v23;
    if ( v10 != v24 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v25 = (unsigned int)v33;
            *(_DWORD *)(v10 + 8) = 0;
            *(_QWORD *)v10 = v10 + 16;
            *(_DWORD *)(v10 + 12) = 4;
            if ( (_DWORD)v25 )
              break;
          }
          v10 += 40;
          if ( v24 == v10 )
            goto LABEL_29;
        }
        v26 = v10;
        v10 += 40;
        sub_1BB9EE0(v26, (__int64)&v32, v25, v7, v8, v9);
      }
      while ( v24 != v10 );
LABEL_29:
      if ( v32 != v34 )
        _libc_free((unsigned __int64)v32);
    }
  }
}
