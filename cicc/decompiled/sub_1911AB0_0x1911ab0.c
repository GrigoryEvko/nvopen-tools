// Function: sub_1911AB0
// Address: 0x1911ab0
//
void __fastcall sub_1911AB0(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rbx
  _BYTE **v11; // rcx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  _DWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r12
  __int64 v25; // r12
  int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // [rsp+0h] [rbp-D0h]
  _BYTE **v30; // [rsp+8h] [rbp-C8h]
  _DWORD *v31; // [rsp+8h] [rbp-C8h]
  _DWORD *v32; // [rsp+18h] [rbp-B8h] BYREF
  int v33; // [rsp+20h] [rbp-B0h]
  char v34; // [rsp+30h] [rbp-A0h]
  char *v35; // [rsp+38h] [rbp-98h]
  __int64 v36; // [rsp+40h] [rbp-90h]
  char v37; // [rsp+48h] [rbp-88h] BYREF
  int v38; // [rsp+60h] [rbp-70h]
  __int64 v39; // [rsp+68h] [rbp-68h]
  char v40; // [rsp+70h] [rbp-60h]
  _BYTE *v41; // [rsp+78h] [rbp-58h] BYREF
  __int64 v42; // [rsp+80h] [rbp-50h]
  _BYTE v43[72]; // [rsp+88h] [rbp-48h] BYREF

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
  v6 = sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = v6;
  v10 = v6;
  if ( v4 )
  {
    v40 = 0;
    v11 = &v41;
    v38 = -1;
    v12 = v4 + (v3 << 6);
    v42 = 0x400000000LL;
    v13 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v41 = v43;
    v14 = v10 + (v13 << 6);
    if ( v10 != v14 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v15 = v38;
            *(_DWORD *)(v10 + 32) = 0;
            *(_DWORD *)(v10 + 36) = 4;
            *(_DWORD *)v10 = v15;
            *(_QWORD *)(v10 + 8) = v39;
            *(_BYTE *)(v10 + 16) = v40;
            *(_QWORD *)(v10 + 24) = v10 + 40;
            if ( (_DWORD)v42 )
              break;
          }
          v10 += 64;
          if ( v14 == v10 )
            goto LABEL_10;
        }
        v16 = v10 + 24;
        v10 += 64;
        v29 = v14;
        v30 = v11;
        sub_1909410(v16, (__int64)v11, (unsigned int)v42, (__int64)v11, v8, v9);
        v14 = v29;
        v11 = v30;
      }
      while ( v29 != v10 );
LABEL_10:
      if ( v41 != v43 )
        _libc_free((unsigned __int64)v41);
    }
    v34 = 0;
    v35 = &v37;
    v33 = -1;
    v36 = 0x400000000LL;
    v38 = -2;
    v40 = 0;
    v41 = v43;
    v42 = 0x400000000LL;
    if ( v12 != v4 )
    {
      v17 = v4;
      do
      {
        if ( *(_DWORD *)v17 <= 0xFFFFFFFD )
        {
          sub_190F0D0(a1, v17, (__int64 *)&v32);
          v19 = v32;
          *v32 = *(_DWORD *)v17;
          v31 = v19;
          *((_QWORD *)v19 + 1) = *(_QWORD *)(v17 + 8);
          v20 = *(unsigned __int8 *)(v17 + 16);
          *((_BYTE *)v19 + 16) = v20;
          sub_19092D0((__int64)(v19 + 6), (char **)(v17 + 24), v20, v21, v22, v23);
          v31[14] = *(_DWORD *)(v17 + 56);
          ++*(_DWORD *)(a1 + 16);
        }
        v18 = *(_QWORD *)(v17 + 24);
        if ( v18 != v17 + 40 )
          _libc_free(v18);
        v17 += 64;
      }
      while ( v12 != v17 );
    }
    j___libc_free_0(v4);
  }
  else
  {
    v40 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    v24 = *(unsigned int *)(a1 + 24);
    v38 = -1;
    v41 = v43;
    v25 = v6 + (v24 << 6);
    v42 = 0x400000000LL;
    if ( v6 != v25 )
    {
      do
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v26 = v38;
            v27 = (unsigned int)v42;
            *(_DWORD *)(v10 + 32) = 0;
            *(_DWORD *)(v10 + 36) = 4;
            *(_DWORD *)v10 = v26;
            *(_QWORD *)(v10 + 8) = v39;
            *(_BYTE *)(v10 + 16) = v40;
            *(_QWORD *)(v10 + 24) = v10 + 40;
            if ( (_DWORD)v27 )
              break;
          }
          v10 += 64;
          if ( v25 == v10 )
            goto LABEL_26;
        }
        v28 = v10 + 24;
        v10 += 64;
        sub_1909410(v28, (__int64)&v41, v7, v27, v8, v9);
      }
      while ( v25 != v10 );
LABEL_26:
      if ( v41 != v43 )
        _libc_free((unsigned __int64)v41);
    }
  }
}
