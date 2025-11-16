// Function: sub_BC0DB0
// Address: 0xbc0db0
//
__int64 __fastcall sub_BC0DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  char v6; // cl
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rbx
  __int64 v12; // r12
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rsi
  char *v18; // rsi
  unsigned int (__fastcall *v19)(_QWORD, _QWORD); // rax
  void **v20; // rax
  __int64 v21; // rcx
  void **v22; // rdx
  __int64 k; // rbx
  __int64 v25; // rdi
  __int64 i; // r13
  __int64 v27; // rdi
  void **v28; // rsi
  __int64 m; // rbx
  __int64 v30; // rdi
  __int64 j; // r13
  __int64 v32; // rdi
  char v33; // [rsp+17h] [rbp-109h]
  _QWORD *v35; // [rsp+20h] [rbp-100h]
  _QWORD *v38; // [rsp+40h] [rbp-E0h]
  _QWORD *v39; // [rsp+48h] [rbp-D8h]
  __int64 v40; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD *v41; // [rsp+58h] [rbp-C8h] BYREF
  _QWORD v42[4]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+80h] [rbp-A0h]
  char v44[8]; // [rsp+90h] [rbp-90h] BYREF
  __int64 v45; // [rsp+98h] [rbp-88h]
  char v46; // [rsp+ACh] [rbp-74h]
  __int64 v47; // [rsp+C8h] [rbp-58h]
  char v48; // [rsp+DCh] [rbp-44h]

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)a1 = 1;
  v5 = sub_BC0510(a4, &unk_4F8A320, a3);
  v6 = *(_BYTE *)(a3 + 872);
  v33 = v6;
  v40 = *(_QWORD *)(v5 + 8);
  if ( LOBYTE(qword_4F80F48[8]) )
  {
    if ( !v6 )
    {
      for ( i = *(_QWORD *)(a3 + 32); a3 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        v27 = i - 56;
        if ( !i )
          v27 = 0;
        sub_B2B950(v27);
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
  }
  else if ( v6 )
  {
    for ( j = *(_QWORD *)(a3 + 32); a3 + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      v32 = j - 56;
      if ( !j )
        v32 = 0;
      sub_B2B9A0(v32);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  sub_C85EE0(v42);
  v42[3] = a3;
  v43 = 0;
  v42[0] = &unk_49DB058;
  v42[2] = &v40;
  v35 = *(_QWORD **)(a2 + 8);
  v39 = *(_QWORD **)a2;
  if ( *(_QWORD **)a2 == v35 )
    goto LABEL_24;
  while ( 1 )
  {
    v43 = *v39;
    if ( (unsigned __int8)sub_BBBED0(&v40, v43, a3) )
      break;
LABEL_23:
    if ( v35 == ++v39 )
      goto LABEL_24;
  }
  (*(void (__fastcall **)(char *, _QWORD, __int64, __int64))(*(_QWORD *)*v39 + 16LL))(v44, *v39, a3, a4);
  sub_BBD520(a4, a3, (__int64)v44, v8);
  if ( v40 )
  {
    v11 = *(_QWORD **)(v40 + 432);
    v38 = &v11[4 * *(unsigned int *)(v40 + 440)];
    if ( v11 != v38 )
    {
      v12 = *v39;
      do
      {
        v41 = 0;
        v13 = (_QWORD *)sub_22077B0(16);
        if ( v13 )
        {
          v13[1] = a3;
          *v13 = &unk_49DB0D8;
        }
        v14 = v41;
        v41 = v13;
        if ( v14 )
          (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
        v15 = v11;
        v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 32LL))(v12);
        if ( (v11[3] & 2) == 0 )
          v15 = (_QWORD *)*v11;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v11[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v15,
          v17,
          v16,
          &v41,
          v44);
        if ( v41 )
          (*(void (__fastcall **)(_QWORD *))(*v41 + 8LL))(v41);
        v11 += 4;
      }
      while ( v38 != v11 );
    }
  }
  v18 = v44;
  sub_BBADB0(a1, (__int64)v44, v9, v10);
  v19 = *(unsigned int (__fastcall **)(_QWORD, _QWORD))(a2 + 24);
  if ( !v19 || (v18 = 0, !v19(*(_QWORD *)(a2 + 32), 0)) )
  {
    if ( !v48 )
      _libc_free(v47, v18);
    if ( !v46 )
      _libc_free(v45, v18);
    goto LABEL_23;
  }
  if ( !v48 )
    _libc_free(v47, 0);
  if ( v46 )
  {
LABEL_24:
    if ( *(_DWORD *)(a1 + 68) == *(_DWORD *)(a1 + 72) )
      goto LABEL_49;
    goto LABEL_25;
  }
  _libc_free(v45, 0);
  if ( *(_DWORD *)(a1 + 68) == *(_DWORD *)(a1 + 72) )
  {
LABEL_49:
    if ( *(_BYTE *)(a1 + 28) )
    {
      v20 = *(void ***)(a1 + 8);
      v28 = &v20[*(unsigned int *)(a1 + 20)];
      LODWORD(v21) = *(_DWORD *)(a1 + 20);
      v22 = v20;
      if ( v20 == v28 )
        goto LABEL_55;
      while ( *v22 != &unk_4F82400 )
      {
        if ( v28 == ++v22 )
        {
LABEL_29:
          while ( *v20 != &unk_4F82428 )
          {
            if ( ++v20 == v22 )
              goto LABEL_55;
          }
          goto LABEL_30;
        }
      }
      goto LABEL_30;
    }
    if ( sub_C8CA60(a1, &unk_4F82400, v7, a1) )
      goto LABEL_30;
  }
LABEL_25:
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_57;
  v20 = *(void ***)(a1 + 8);
  v21 = *(unsigned int *)(a1 + 20);
  v22 = &v20[v21];
  if ( v22 != v20 )
    goto LABEL_29;
LABEL_55:
  if ( (unsigned int)v21 < *(_DWORD *)(a1 + 16) )
  {
    *(_DWORD *)(a1 + 20) = v21 + 1;
    *v22 = &unk_4F82428;
    ++*(_QWORD *)a1;
  }
  else
  {
LABEL_57:
    sub_C8CC70(a1, &unk_4F82428);
  }
LABEL_30:
  v42[0] = &unk_49DB058;
  nullsub_162(v42);
  if ( v33 )
  {
    if ( !*(_BYTE *)(a3 + 872) )
    {
      for ( k = *(_QWORD *)(a3 + 32); a3 + 24 != k; k = *(_QWORD *)(k + 8) )
      {
        v25 = k - 56;
        if ( !k )
          v25 = 0;
        sub_B2B950(v25);
      }
      *(_BYTE *)(a3 + 872) = 1;
    }
  }
  else if ( *(_BYTE *)(a3 + 872) )
  {
    for ( m = *(_QWORD *)(a3 + 32); a3 + 24 != m; m = *(_QWORD *)(m + 8) )
    {
      v30 = m - 56;
      if ( !m )
        v30 = 0;
      sub_B2B9A0(v30);
    }
    *(_BYTE *)(a3 + 872) = 0;
  }
  return a1;
}
