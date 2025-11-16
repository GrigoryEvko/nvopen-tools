// Function: sub_138F7D0
// Address: 0x138f7d0
//
__int64 __fastcall sub_138F7D0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v5; // rcx
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v8; // r14
  unsigned int v9; // edx
  int *v10; // rdi
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // r13
  unsigned int v15; // edx
  int *v16; // rdi
  int v17; // edx
  unsigned int v18; // r10d
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r9
  unsigned int v24; // eax
  unsigned int v25; // edx
  int *v26; // r11
  int v27; // edx
  __int64 *v28; // rdi
  unsigned int v29; // r12d
  unsigned int v31; // eax
  __int64 *v32; // r8
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r9
  __int64 v36; // rcx
  unsigned int v37; // eax
  __int64 v38; // rdx
  unsigned int v39; // edx
  int *v40; // r10
  int v41; // r10d
  unsigned int v42; // [rsp+Ch] [rbp-94h]
  __int64 *v43; // [rsp+20h] [rbp-80h] BYREF
  __int64 v44; // [rsp+28h] [rbp-78h]
  _BYTE v45[112]; // [rsp+30h] [rbp-70h] BYREF

  v5 = *(_QWORD *)(a1 + 32);
  v6 = v5 + 32LL * a2;
  v7 = *(_DWORD *)(v6 + 24);
  v8 = v6;
  if ( v7 != -1 )
  {
    v9 = *(_DWORD *)(v6 + 24);
    do
    {
      v10 = (int *)(v5 + 32LL * v9);
      v9 = v10[6];
    }
    while ( v9 != -1 );
    v11 = *v10;
    while ( 1 )
    {
      *(_DWORD *)(v6 + 24) = v11;
      v8 = v5 + 32LL * v7;
      v5 = *(_QWORD *)(a1 + 32);
      v7 = *(_DWORD *)(v8 + 24);
      if ( v7 == -1 )
        break;
      v6 = v8;
    }
  }
  v12 = v5 + 32LL * a3;
  v13 = *(_DWORD *)(v12 + 24);
  v14 = v12;
  if ( v13 != -1 )
  {
    v15 = *(_DWORD *)(v12 + 24);
    do
    {
      v16 = (int *)(v5 + 32LL * v15);
      v15 = v16[6];
    }
    while ( v15 != -1 );
    v17 = *v16;
    while ( 1 )
    {
      *(_DWORD *)(v12 + 24) = v17;
      v14 = v5 + 32LL * v13;
      v13 = *(_DWORD *)(v14 + 24);
      if ( v13 == -1 )
        break;
      v5 = *(_QWORD *)(a1 + 32);
      v12 = v14;
    }
  }
  if ( v14 == v8 )
  {
    return 1;
  }
  else
  {
    v18 = 0;
    v19 = *(_QWORD *)(v8 + 16);
    v20 = v8;
    v43 = (__int64 *)v45;
    v44 = 0x800000000LL;
    v21 = (__int64 *)v45;
    if ( *(_DWORD *)(v8 + 8) == -1 )
    {
      return 0;
    }
    else
    {
      while ( 1 )
      {
        v21[v18] = v20;
        v22 = *(_QWORD *)(a1 + 32);
        v19 |= *(_QWORD *)(v20 + 16);
        v18 = v44 + 1;
        LODWORD(v44) = v44 + 1;
        v23 = v22 + 32LL * *(unsigned int *)(v20 + 8);
        v24 = *(_DWORD *)(v23 + 24);
        v20 = v23;
        if ( v24 != -1 )
        {
          v25 = *(_DWORD *)(v23 + 24);
          do
          {
            v26 = (int *)(v22 + 32LL * v25);
            v25 = v26[6];
          }
          while ( v25 != -1 );
          v27 = *v26;
          while ( 1 )
          {
            *(_DWORD *)(v23 + 24) = v27;
            v20 = v22 + 32LL * v24;
            v24 = *(_DWORD *)(v20 + 24);
            if ( v24 == -1 )
              break;
            v22 = *(_QWORD *)(a1 + 32);
            v23 = v20;
          }
        }
        if ( *(_DWORD *)(v20 + 8) == -1 )
          break;
        if ( v20 == v14 )
        {
          v28 = v43;
          goto LABEL_32;
        }
        if ( HIDWORD(v44) <= v18 )
        {
          v42 = a3;
          sub_16CD150(&v43, v45, 0, 8);
          v18 = v44;
          a3 = v42;
        }
        v21 = v43;
      }
      v28 = v43;
      if ( v20 != v14 )
      {
        v29 = 0;
        goto LABEL_28;
      }
LABEL_32:
      *(_QWORD *)(v14 + 16) |= v19;
      v31 = *(_DWORD *)(v8 + 12);
      if ( v31 == -1 )
      {
        *(_DWORD *)(v14 + 12) = -1;
      }
      else
      {
        *(_DWORD *)(v14 + 12) = v31;
        v35 = *(_QWORD *)(a1 + 32);
        v36 = v35 + 32LL * v31;
        v37 = *(_DWORD *)(v36 + 24);
        v38 = v36;
        if ( v37 != -1 )
        {
          v39 = *(_DWORD *)(v36 + 24);
          do
          {
            v40 = (int *)(v35 + 32LL * v39);
            v39 = v40[6];
          }
          while ( v39 != -1 );
          v41 = *v40;
          while ( 1 )
          {
            *(_DWORD *)(v36 + 24) = v41;
            v38 = v35 + 32LL * v37;
            v37 = *(_DWORD *)(v38 + 24);
            if ( v37 == -1 )
              break;
            v35 = *(_QWORD *)(a1 + 32);
            v36 = v38;
          }
        }
        *(_DWORD *)(v38 + 8) = a3;
        v18 = v44;
      }
      v32 = &v28[v18];
      if ( v28 != v32 )
      {
        v33 = v28;
        do
        {
          v34 = *v33++;
          *(_DWORD *)(v34 + 24) = *(_DWORD *)v14;
        }
        while ( v32 != v33 );
      }
      v29 = 1;
LABEL_28:
      if ( v28 != (__int64 *)v45 )
        _libc_free((unsigned __int64)v28);
    }
  }
  return v29;
}
