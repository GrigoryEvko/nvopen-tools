// Function: sub_391AF60
// Address: 0x391af60
//
void __fastcall sub_391AF60(__int64 a1, int a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r15
  unsigned int *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rdx
  int v15; // edx
  int v16; // esi
  __int64 v17; // rdi
  unsigned int *v18; // r14
  _DWORD *v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // r8d
  int v27; // r9d
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // r12
  int v32; // edi
  __int64 v33; // rdx
  int v34; // edx
  int v35; // r8d
  __int64 v36; // rdi
  _DWORD *v37; // [rsp+28h] [rbp-B8h] BYREF
  int v38; // [rsp+30h] [rbp-B0h]
  char *v39; // [rsp+38h] [rbp-A8h]
  __int64 v40; // [rsp+40h] [rbp-A0h]
  char v41; // [rsp+48h] [rbp-98h] BYREF
  char *v42; // [rsp+50h] [rbp-90h]
  __int64 v43; // [rsp+58h] [rbp-88h]
  char v44; // [rsp+60h] [rbp-80h] BYREF
  int i; // [rsp+70h] [rbp-70h]
  char *v46; // [rsp+78h] [rbp-68h] BYREF
  __int64 v47; // [rsp+80h] [rbp-60h]
  char v48; // [rsp+88h] [rbp-58h] BYREF
  _BYTE *v49; // [rsp+90h] [rbp-50h] BYREF
  __int64 v50; // [rsp+98h] [rbp-48h]
  _BYTE v51[64]; // [rsp+A0h] [rbp-40h] BYREF

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
  v6 = sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  v10 = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v11 = (unsigned int *)(v4 + 72 * v3);
    v12 = *(unsigned int *)(a1 + 24);
    v46 = &v48;
    v47 = 0x100000000LL;
    v13 = v6 + 72 * v12;
    v49 = v51;
    v50 = 0x400000000LL;
    i = 1;
    if ( v6 != v13 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v15 = i;
            v16 = v47;
            *(_DWORD *)(v10 + 16) = 0;
            *(_DWORD *)(v10 + 20) = 1;
            *(_DWORD *)v10 = v15;
            *(_QWORD *)(v10 + 8) = v10 + 24;
            if ( v16 )
              sub_39199E0(v10 + 8, (__int64)&v46, v10 + 24, v7, v8, v9);
            v7 = (unsigned int)v50;
            v14 = v10 + 48;
            *(_DWORD *)(v10 + 40) = 0;
            *(_QWORD *)(v10 + 32) = v10 + 48;
            *(_DWORD *)(v10 + 44) = 4;
            if ( (_DWORD)v7 )
              break;
          }
          v10 += 72;
          if ( v13 == v10 )
            goto LABEL_12;
        }
        v17 = v10 + 32;
        v10 += 72;
        sub_39199E0(v17, (__int64)&v49, v14, v7, v8, v9);
      }
      while ( v13 != v10 );
LABEL_12:
      if ( v49 != v51 )
        _libc_free((unsigned __int64)v49);
      if ( v46 != &v48 )
        _libc_free((unsigned __int64)v46);
    }
    v18 = (unsigned int *)v4;
    v39 = &v41;
    v42 = &v44;
    v46 = &v48;
    v43 = 0x400000000LL;
    v50 = 0x400000000LL;
    v40 = 0x100000000LL;
    v38 = 1;
    v47 = 0x100000000LL;
    v49 = v51;
    for ( i = 2; v11 != v18; v18 += 18 )
    {
      if ( *v18 != 1 && *v18 != 2 || v18[4] || v18[10] )
      {
        sub_391AD00(a1, (int *)v18, &v37);
        v19 = v37;
        v20 = *v18;
        *v37 = v20;
        sub_3919AC0((__int64)(v19 + 2), (char **)v18 + 1, v20, v21, v22, v23);
        sub_3919AC0((__int64)(v19 + 8), (char **)v18 + 4, v24, v25, v26, v27);
        v19[16] = v18[16];
        ++*(_DWORD *)(a1 + 16);
      }
      v28 = *((_QWORD *)v18 + 4);
      if ( (unsigned int *)v28 != v18 + 12 )
        _libc_free(v28);
      v29 = *((_QWORD *)v18 + 1);
      if ( (unsigned int *)v29 != v18 + 6 )
        _libc_free(v29);
    }
    j___libc_free_0(v4);
  }
  else
  {
    v30 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v46 = &v48;
    v47 = 0x100000000LL;
    v31 = v6 + 72 * v30;
    v49 = v51;
    v50 = 0x400000000LL;
    i = 1;
    if ( v6 != v31 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v34 = i;
            v35 = v47;
            *(_DWORD *)(v10 + 16) = 0;
            *(_DWORD *)(v10 + 20) = 1;
            *(_DWORD *)v10 = v34;
            *(_QWORD *)(v10 + 8) = v10 + 24;
            if ( v35 )
              sub_39199E0(v10 + 8, (__int64)&v46, v10 + 24, v7, v35, v9);
            v32 = v50;
            v33 = v10 + 48;
            *(_DWORD *)(v10 + 40) = 0;
            *(_QWORD *)(v10 + 32) = v10 + 48;
            *(_DWORD *)(v10 + 44) = 4;
            if ( v32 )
              break;
          }
          v10 += 72;
          if ( v31 == v10 )
            goto LABEL_38;
        }
        v36 = v10 + 32;
        v10 += 72;
        sub_39199E0(v36, (__int64)&v49, v33, v7, v35, v9);
      }
      while ( v31 != v10 );
LABEL_38:
      if ( v49 != v51 )
        _libc_free((unsigned __int64)v49);
      if ( v46 != &v48 )
        _libc_free((unsigned __int64)v46);
    }
  }
}
