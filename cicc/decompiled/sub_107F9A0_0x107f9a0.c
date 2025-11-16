// Function: sub_107F9A0
// Address: 0x107f9a0
//
void __fastcall sub_107F9A0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r14
  _QWORD *v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  _QWORD *v28; // rdi
  int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // r12
  int v32; // edi
  __int64 v33; // r8
  __int64 v34; // [rsp+0h] [rbp-E0h]
  __int64 v35[3]; // [rsp+28h] [rbp-B8h] BYREF
  char v36; // [rsp+40h] [rbp-A0h] BYREF
  char *v37; // [rsp+48h] [rbp-98h]
  __int64 v38; // [rsp+50h] [rbp-90h]
  char v39; // [rsp+58h] [rbp-88h] BYREF
  __int64 v40; // [rsp+68h] [rbp-78h]
  char *v41; // [rsp+70h] [rbp-70h] BYREF
  __int64 v42; // [rsp+78h] [rbp-68h]
  char v43; // [rsp+80h] [rbp-60h] BYREF
  _BYTE *v44; // [rsp+88h] [rbp-58h] BYREF
  __int64 i; // [rsp+90h] [rbp-50h]
  _BYTE v46[16]; // [rsp+98h] [rbp-48h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v3 = 8;
  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v7 < 0x40 )
    v7 = 64;
  *(_DWORD *)(a1 + 24) = v7;
  v8 = sub_C7D670(72LL * v7, 8);
  *(_QWORD *)(a1 + 8) = v8;
  v12 = v8;
  if ( v6 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v13 = 72 * v5;
    v42 = 0x100000000LL;
    v14 = (_QWORD *)(v6 + 72 * v5);
    v47 = 0x100000000LL;
    v34 = v13;
    v15 = 9LL * *(unsigned int *)(a1 + 24);
    v41 = &v43;
    v16 = v12 + 8 * v15;
    v44 = v46;
    i = 0x400000000LL;
    if ( v12 != v16 )
    {
      do
      {
        if ( v12 )
        {
          v3 = (unsigned int)v42;
          *(_DWORD *)(v12 + 8) = 0;
          *(_QWORD *)v12 = v12 + 16;
          *(_DWORD *)(v12 + 12) = 1;
          if ( (_DWORD)v3 )
          {
            v3 = (__int64)&v41;
            sub_10774E0(v12, (__int64)&v41, v12 + 16, v9, v10, v11);
          }
          v9 = (unsigned int)i;
          *(_DWORD *)(v12 + 32) = 0;
          *(_QWORD *)(v12 + 24) = v12 + 40;
          *(_DWORD *)(v12 + 36) = 4;
          if ( (_DWORD)v9 )
          {
            v3 = (__int64)&v44;
            sub_10774E0(v12 + 24, (__int64)&v44, v12 + 40, v9, v10, v11);
          }
          *(_QWORD *)(v12 + 56) = v47;
        }
        v12 += 72;
      }
      while ( v16 != v12 );
      if ( v44 != v46 )
        _libc_free(v44, v3);
      if ( v41 != &v43 )
        _libc_free(v41, v3);
    }
    v17 = (_QWORD *)v6;
    v35[1] = (__int64)&v36;
    v35[2] = 0x100000000LL;
    v40 = 0x100000000LL;
    v42 = 0x100000000LL;
    v37 = &v39;
    v44 = v46;
    v47 = 0x200000000LL;
    v38 = 0x400000000LL;
    v41 = &v43;
    for ( i = 0x400000000LL; v14 != v17; v17 += 9 )
    {
      v29 = *((_DWORD *)v17 + 15);
      if ( v29 != 1 && v29 != 2 || *((_DWORD *)v17 + 2) || *((_DWORD *)v17 + 8) )
      {
        sub_107D670(a1, (__int64)v17, v35);
        v18 = v35[0];
        sub_1077380(v35[0], (char **)v17, v19, v20, v21, v22);
        v3 = (__int64)(v17 + 3);
        sub_1077380(v18 + 24, (char **)v17 + 3, v23, v24, v25, v26);
        v27 = v35[0];
        *(_DWORD *)(v18 + 56) = *((_DWORD *)v17 + 14);
        *(_DWORD *)(v18 + 60) = *((_DWORD *)v17 + 15);
        *(_DWORD *)(v27 + 64) = *((_DWORD *)v17 + 16);
        ++*(_DWORD *)(a1 + 16);
      }
      v28 = (_QWORD *)v17[3];
      if ( v28 != v17 + 5 )
        _libc_free(v28, v3);
      if ( (_QWORD *)*v17 != v17 + 2 )
        _libc_free(*v17, v3);
    }
    sub_C7D6A0(v6, v34, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v42 = 0x100000000LL;
    v47 = 0x100000000LL;
    v30 = *(unsigned int *)(a1 + 24);
    v41 = &v43;
    v31 = v8 + 72 * v30;
    v44 = v46;
    i = 0x400000000LL;
    if ( v8 != v31 )
    {
      do
      {
        if ( v12 )
        {
          v33 = (unsigned int)v42;
          *(_DWORD *)(v12 + 8) = 0;
          *(_QWORD *)v12 = v12 + 16;
          *(_DWORD *)(v12 + 12) = 1;
          if ( (_DWORD)v33 )
          {
            v3 = (__int64)&v41;
            sub_10774E0(v12, (__int64)&v41, v12 + 16, v9, v33, v11);
          }
          v32 = i;
          *(_DWORD *)(v12 + 32) = 0;
          *(_QWORD *)(v12 + 24) = v12 + 40;
          *(_DWORD *)(v12 + 36) = 4;
          if ( v32 )
          {
            v3 = (__int64)&v44;
            sub_10774E0(v12 + 24, (__int64)&v44, v12 + 40, v9, v33, v11);
          }
          *(_QWORD *)(v12 + 56) = v47;
        }
        v12 += 72;
      }
      while ( v31 != v12 );
      if ( v44 != v46 )
        _libc_free(v44, v3);
      if ( v41 != &v43 )
        _libc_free(v41, v3);
    }
  }
}
