// Function: sub_19A0530
// Address: 0x19a0530
//
void __fastcall sub_19A0530(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rbx
  _QWORD **v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  char *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-B0h]
  _QWORD **v28; // [rsp+8h] [rbp-A8h]
  __int64 v29[3]; // [rsp+18h] [rbp-98h] BYREF
  __int64 v30; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v31; // [rsp+50h] [rbp-60h] BYREF
  __int64 v32; // [rsp+58h] [rbp-58h]
  _QWORD v33[10]; // [rsp+60h] [rbp-50h] BYREF

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
  v6 = sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  v10 = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v32 = 0x400000001LL;
    v11 = &v31;
    v12 = *(unsigned int *)(a1 + 24);
    v31 = v33;
    v13 = v4 + 48 * v3;
    v33[0] = -1;
    v14 = v10 + 48 * v12;
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
            v15 = (unsigned int)v32;
            *(_DWORD *)(v10 + 12) = 4;
            if ( (_DWORD)v15 )
              break;
          }
          v10 += 48;
          if ( v14 == v10 )
            goto LABEL_10;
        }
        v16 = v10;
        v27 = v14;
        v10 += 48;
        v28 = v11;
        sub_19930D0(v16, (__int64)v11, v15, (__int64)v11, v8, v9);
        v14 = v27;
        v11 = v28;
      }
      while ( v27 != v10 );
LABEL_10:
      if ( v31 != v33 )
        _libc_free((unsigned __int64)v31);
    }
    v31 = v33;
    v17 = v4;
    v29[1] = (__int64)&v30;
    v29[2] = 0x400000001LL;
    v30 = -1;
    v32 = 0x400000001LL;
    for ( v33[0] = -2; v13 != v17; v17 += 48 )
    {
      if ( *(_DWORD *)(v17 + 8) != 1 || (v22 = *(char **)v17, **(_QWORD **)v17 != v30) && v33[0] != *(_QWORD *)v22 )
      {
        sub_19A0320(a1, v17, v29);
        sub_19931B0(v29[0], (char **)v17, v18, v19, v20, v21);
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
    v33[0] = -1;
    v32 = 0x400000001LL;
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v31 = v33;
    v24 = v10 + 48 * v23;
    if ( v10 != v24 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v25 = (unsigned int)v32;
            *(_DWORD *)(v10 + 8) = 0;
            *(_QWORD *)v10 = v10 + 16;
            *(_DWORD *)(v10 + 12) = 4;
            if ( (_DWORD)v25 )
              break;
          }
          v10 += 48;
          if ( v24 == v10 )
            goto LABEL_29;
        }
        v26 = v10;
        v10 += 48;
        sub_19930D0(v26, (__int64)&v31, v7, v25, v8, v9);
      }
      while ( v24 != v10 );
LABEL_29:
      if ( v31 != v33 )
        _libc_free((unsigned __int64)v31);
    }
  }
}
