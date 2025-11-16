// Function: sub_34BD420
// Address: 0x34bd420
//
void __fastcall sub_34BD420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rbx
  _QWORD *v10; // rax
  unsigned __int64 v11; // r15
  _BYTE *v12; // rdx
  _QWORD *i; // rdx
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // r13
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // r10
  __int64 v24; // r15
  _QWORD *v25; // rax
  _QWORD *v26; // r15
  _BYTE *v27; // rbx
  _BYTE *v28; // r12
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // rdi
  _BYTE *v31; // rax
  _BYTE *v32; // r15
  unsigned __int64 v33; // r10
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // [rsp+8h] [rbp-98h]
  _BYTE *v36; // [rsp+20h] [rbp-80h]
  _BYTE *v37; // [rsp+20h] [rbp-80h]
  unsigned int v38; // [rsp+2Ch] [rbp-74h]
  int v39; // [rsp+2Ch] [rbp-74h]
  unsigned int v40; // [rsp+2Ch] [rbp-74h]
  _BYTE *v41; // [rsp+30h] [rbp-70h] BYREF
  __int64 v42; // [rsp+38h] [rbp-68h]
  _BYTE v43[96]; // [rsp+40h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(a1 + 128);
  *(_DWORD *)(a1 + 144) = *(_DWORD *)(v7 + 120);
  v8 = *(_QWORD *)(v7 + 104) - *(_QWORD *)(v7 + 96);
  v42 = 0x600000000LL;
  v9 = v8 >> 3;
  v10 = v43;
  v11 = (unsigned int)(v9 + 1);
  v41 = v43;
  if ( (_DWORD)v9 != -1 )
  {
    v12 = v43;
    if ( v11 > 6 )
    {
      sub_239B9C0((__int64)&v41, v11, (__int64)v43, a4, a5, a6);
      v12 = v41;
      v10 = &v41[8 * (unsigned int)v42];
    }
    for ( i = &v12[8 * v11]; i != v10; ++v10 )
    {
      if ( v10 )
        *v10 = 0;
    }
    LODWORD(v42) = v9 + 1;
  }
  v14 = *(_QWORD *)(a1 + 48);
  v15 = (__int64 *)(v14 + 8LL * *(unsigned int *)(a1 + 56));
  v16 = (__int64 *)v14;
  if ( (__int64 *)v14 != v15 )
  {
    while ( 1 )
    {
      v19 = *v16;
      if ( *v16 )
        break;
LABEL_16:
      if ( v15 == ++v16 )
        goto LABEL_32;
    }
    if ( *(_QWORD *)v19 )
    {
      v20 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v19 + 24LL) + 1);
      v21 = 8 * v20;
    }
    else
    {
      v21 = 0;
      LODWORD(v20) = 0;
    }
    v22 = (unsigned int)v42;
    if ( (unsigned int)v42 > (unsigned int)v20 || (v23 = (unsigned int)(v20 + 1), a5 = v23, v23 == (unsigned int)v42) )
    {
      a4 = (__int64)v41;
      goto LABEL_12;
    }
    v24 = 8 * v23;
    if ( v23 < (unsigned int)v42 )
    {
      a4 = (__int64)v41;
      v31 = &v41[8 * (unsigned int)v42];
      v32 = &v41[v24];
      if ( v31 == v32 )
        goto LABEL_30;
      do
      {
        v33 = *((_QWORD *)v31 - 1);
        v31 -= 8;
        if ( v33 )
        {
          v34 = *(_QWORD *)(v33 + 24);
          if ( v34 != v33 + 40 )
          {
            v35 = v33;
            v36 = v31;
            v39 = a5;
            _libc_free(v34);
            v33 = v35;
            v31 = v36;
            LODWORD(a5) = v39;
          }
          v37 = v31;
          v40 = a5;
          j_j___libc_free_0(v33);
          v31 = v37;
          a5 = v40;
        }
      }
      while ( v32 != v31 );
    }
    else
    {
      if ( v23 > HIDWORD(v42) )
      {
        v38 = v20 + 1;
        sub_239B9C0((__int64)&v41, v23, v14, HIDWORD(v42), v23, a6);
        v22 = (unsigned int)v42;
        a5 = v38;
      }
      a4 = (__int64)v41;
      v25 = &v41[8 * v22];
      v26 = &v41[v24];
      if ( v25 == v26 )
        goto LABEL_30;
      do
      {
        if ( v25 )
          *v25 = 0;
        ++v25;
      }
      while ( v26 != v25 );
    }
    a4 = (__int64)v41;
LABEL_30:
    LODWORD(v42) = a5;
    v19 = *v16;
LABEL_12:
    *v16 = 0;
    v17 = *(_QWORD *)(a4 + v21);
    *(_QWORD *)(a4 + v21) = v19;
    if ( v17 )
    {
      v18 = *(_QWORD *)(v17 + 24);
      if ( v18 != v17 + 40 )
        _libc_free(v18);
      j_j___libc_free_0(v17);
    }
    goto LABEL_16;
  }
LABEL_32:
  sub_34BD060(a1 + 48, (__int64)&v41, v14, a4, a5);
  v27 = v41;
  v28 = &v41[8 * (unsigned int)v42];
  if ( v41 != v28 )
  {
    do
    {
      v29 = *((_QWORD *)v28 - 1);
      v28 -= 8;
      if ( v29 )
      {
        v30 = *(_QWORD *)(v29 + 24);
        if ( v30 != v29 + 40 )
          _libc_free(v30);
        j_j___libc_free_0(v29);
      }
    }
    while ( v27 != v28 );
    v28 = v41;
  }
  if ( v28 != v43 )
    _libc_free((unsigned __int64)v28);
}
