// Function: sub_B89820
// Address: 0xb89820
//
__int64 __fastcall sub_B89820(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 *v5; // r13
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // r13d
  __int64 k; // r12
  __int64 v15; // rdi
  unsigned int m; // r12d
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r8
  _QWORD *v27; // rax
  _QWORD *v28; // rsi
  unsigned int (__fastcall *v29)(_QWORD, _QWORD); // rax
  unsigned int v30; // r12d
  int v31; // r12d
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // r12
  __int64 n; // rbx
  __int64 v36; // r14
  __int64 v37; // r8
  __int64 v38; // r12
  __int64 v39; // rbx
  _QWORD *v40; // rdi
  unsigned int (__fastcall *v41)(_QWORD, _QWORD); // rax
  __int64 *v42; // rbx
  __int64 *v43; // r12
  int v44; // r14d
  __int64 v45; // rdi
  __int64 (*v46)(); // rax
  char v47; // al
  unsigned int v49; // eax
  unsigned int v50; // r8d
  __int64 v51; // rcx
  __int64 ii; // rbx
  __int64 v53; // rdi
  __int64 i; // r14
  __int64 v55; // rdi
  __int64 jj; // rbx
  __int64 v57; // rdi
  __int64 j; // r14
  __int64 v59; // rdi
  char v60; // [rsp+7h] [rbp-A9h]
  __int64 v61; // [rsp+8h] [rbp-A8h]
  unsigned int v63; // [rsp+18h] [rbp-98h]
  unsigned int v64; // [rsp+1Ch] [rbp-94h]
  unsigned __int8 v65; // [rsp+28h] [rbp-88h]
  unsigned __int8 v66; // [rsp+29h] [rbp-87h]
  char v67; // [rsp+2Ah] [rbp-86h]
  unsigned __int8 v68; // [rsp+2Bh] [rbp-85h]
  unsigned int v69; // [rsp+2Ch] [rbp-84h]
  __int64 v70; // [rsp+30h] [rbp-80h] BYREF
  __int64 v71; // [rsp+38h] [rbp-78h]
  __int64 v72; // [rsp+40h] [rbp-70h]
  _QWORD v73[12]; // [rsp+50h] [rbp-60h] BYREF

  v2 = a2;
  v3 = a1 + 568;
  sub_B85E70(a1 + 568);
  sub_B80B80(a1 + 568);
  v60 = *(_BYTE *)(a2 + 872);
  if ( LOBYTE(qword_4F80F48[8]) )
  {
    if ( !*(_BYTE *)(a2 + 872) )
    {
      for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        v55 = i - 56;
        if ( !i )
          v55 = 0;
        sub_B2B950(v55);
      }
      *(_BYTE *)(a2 + 872) = 1;
    }
  }
  else if ( v60 )
  {
    for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      v59 = j - 56;
      if ( !j )
        v59 = 0;
      sub_B2B9A0(v59);
    }
    *(_BYTE *)(a2 + 872) = 0;
  }
  v65 = 0;
  v4 = *(__int64 **)(a1 + 824);
  v5 = &v4[*(unsigned int *)(a1 + 832)];
  if ( v4 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = *v4;
        v7 = *(__int64 (**)())(*(_QWORD *)*v4 + 24LL);
        if ( v7 != sub_97DD00 )
          break;
        if ( v5 == ++v4 )
          goto LABEL_8;
      }
      ++v4;
      v65 |= ((__int64 (__fastcall *)(__int64, unsigned __int64))v7)(v6, a2);
    }
    while ( v5 != v4 );
LABEL_8:
    v2 = a2;
  }
  sub_B80C30(v3);
  if ( !*(_DWORD *)(a1 + 608) )
    goto LABEL_57;
  v63 = 0;
  do
  {
    v8 = *(_QWORD *)(v2 + 176);
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v63);
    v10 = *(_QWORD *)(v2 + 168);
    if ( !v9 )
    {
      sub_C996C0("OptModule", 9, v10, v8);
      BUG();
    }
    v11 = sub_C996C0("OptModule", 9, v10, v8);
    v12 = *(_QWORD *)(v9 + 424);
    v13 = 0;
    v61 = v11;
    for ( k = v12 + 16LL * *(unsigned int *)(v9 + 432);
          k != v12;
          v13 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v15 + 24LL))(v15, v2) )
    {
      v15 = *(_QWORD *)(v12 + 8);
      v12 += 16;
    }
    for ( m = 0;
          *(_DWORD *)(v9 + 24) > m;
          v13 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v18 + 24LL))(v18, v2) )
    {
      v17 = m++;
      v18 = *(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v17);
    }
    v19 = *(_QWORD *)v2;
    v70 = 0;
    v71 = 0;
    v72 = 0x1000000000LL;
    v20 = sub_B6F970(v19);
    v67 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v20 + 24LL))(v20, "size-info", 9);
    if ( v67 )
      v64 = sub_B806A0(v9, v2, (__int64)&v70);
    v21 = *(unsigned int *)(v9 + 24);
    if ( !(_DWORD)v21 )
      goto LABEL_43;
    v66 = v13;
    v69 = 0;
    while ( 1 )
    {
      v22 = *(_QWORD *)(*(_QWORD *)(v9 + 16) + 8LL * v69);
      sub_B817B0(v9, v22, 0, 4, *(const void **)(v2 + 168), *(_QWORD *)(v2 + 176));
      sub_B86470(v9, (__int64 *)v22);
      sub_B89740(v9, (__int64 *)v22);
      sub_C85EE0(v73);
      v73[2] = v22;
      v73[3] = 0;
      v73[0] = &unk_49DA748;
      v73[4] = v2;
      v23 = sub_BC4450(v22);
      v24 = v23;
      if ( v23 )
        sub_C9E250(v23);
      if ( *(_BYTE *)(v22 + 168) )
      {
        v25 = *(_QWORD *)(v22 + 120);
        v26 = *(_QWORD **)(*(_QWORD *)(v22 + 112) + 8 * (v2 % v25));
        if ( !v26 )
          goto LABEL_69;
        v27 = (_QWORD *)*v26;
        if ( v2 != *(_QWORD *)(*v26 + 8LL) )
        {
          while ( 1 )
          {
            v28 = (_QWORD *)*v27;
            if ( !*v27 )
              break;
            v26 = v27;
            if ( v2 % v25 != v28[1] % v25 )
              break;
            v27 = (_QWORD *)*v27;
            if ( v2 == v28[1] )
              goto LABEL_28;
          }
LABEL_69:
          v68 = 0;
          goto LABEL_31;
        }
LABEL_28:
        if ( !*v26 )
          goto LABEL_69;
      }
      v68 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v22 + 144LL))(v22, v2);
      if ( v68 )
      {
        sub_B80140(v22, v2);
        v66 = v68;
      }
LABEL_31:
      if ( v67 )
      {
        v49 = sub_BAA3C0(v2);
        v50 = v64;
        if ( v49 != v64 )
        {
          v51 = v49 - (unsigned __int64)v64;
          v64 = v49;
          sub_B82CC0(v9, v22, v2, v51, v50, (__int64)&v70, 0);
        }
      }
      if ( v24 )
        sub_C9E2A0(v24);
      v73[0] = &unk_49DA748;
      nullsub_162(v73);
      if ( v68 )
      {
        sub_B817B0(v9, v22, 1, 4, *(const void **)(v2 + 168), *(_QWORD *)(v2 + 176));
        sub_B865A0(v9, (__int64 *)v22);
        sub_B866C0(v9, (__int64 *)v22);
        nullsub_76();
        sub_B887D0(v9, (__int64 *)v22);
      }
      else
      {
        sub_B865A0(v9, (__int64 *)v22);
        sub_B866C0(v9, (__int64 *)v22);
        nullsub_76();
      }
      sub_B87180(v9, v22);
      v21 = v22;
      sub_B81BF0(v9, v22, *(const void **)(v2 + 168), *(_QWORD *)(v2 + 176), 4);
      v29 = *(unsigned int (__fastcall **)(_QWORD, _QWORD))(v9 + 440);
      if ( v29 )
      {
        v21 = 0;
        if ( v29(*(_QWORD *)(v9 + 448), 0) )
          break;
      }
      ++v69;
      v30 = *(_DWORD *)(v9 + 24);
      if ( v30 <= v69 )
      {
        v13 = v66;
        goto LABEL_40;
      }
    }
    v13 = v66;
    v30 = *(_DWORD *)(v9 + 24);
LABEL_40:
    v31 = v30 - 1;
    if ( v31 >= 0 )
    {
      v32 = 8LL * v31;
      do
      {
        v21 = v2;
        --v31;
        v33 = *(_QWORD *)(*(_QWORD *)(v9 + 16) + v32);
        v32 -= 8;
        v13 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v33 + 32LL))(v33, v2);
      }
      while ( v31 != -1 );
    }
LABEL_43:
    v34 = *(_QWORD *)(v9 + 424);
    for ( n = v34 + 16LL * *(unsigned int *)(v9 + 432);
          n != v34;
          v13 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v36 + 32LL))(v36, v2) )
    {
      v36 = *(_QWORD *)(v34 + 8);
      v34 += 16;
      sub_B808B0(v36);
      v21 = v2;
    }
    v37 = v70;
    if ( HIDWORD(v71) && (_DWORD)v71 )
    {
      v38 = 8LL * (unsigned int)v71;
      v39 = 0;
      do
      {
        v40 = *(_QWORD **)(v37 + v39);
        if ( v40 && v40 != (_QWORD *)-8LL )
        {
          v21 = *v40 + 17LL;
          sub_C7D6A0(v40, v21, 8);
          v37 = v70;
        }
        v39 += 8;
      }
      while ( v38 != v39 );
    }
    _libc_free(v37, v21);
    if ( v61 )
      sub_C9AF60(v61);
    v65 |= v13;
    sub_B6EAA0(*(_QWORD *)v2);
    v41 = *(unsigned int (__fastcall **)(_QWORD, _QWORD))(a1 + 1288);
    if ( v41 && v41(*(_QWORD *)(a1 + 1296), 0) )
      break;
    ++v63;
  }
  while ( v63 < *(_DWORD *)(a1 + 608) );
LABEL_57:
  v42 = *(__int64 **)(a1 + 824);
  v43 = &v42[*(unsigned int *)(a1 + 832)];
  if ( v43 != v42 )
  {
    v44 = v65;
    do
    {
      while ( 1 )
      {
        v45 = *v42;
        v46 = *(__int64 (**)())(*(_QWORD *)*v42 + 32LL);
        if ( v46 != sub_97DD10 )
          break;
        if ( v43 == ++v42 )
          goto LABEL_62;
      }
      ++v42;
      v44 |= ((__int64 (__fastcall *)(__int64, unsigned __int64))v46)(v45, v2);
    }
    while ( v43 != v42 );
LABEL_62:
    v65 = v44;
  }
  v47 = *(_BYTE *)(v2 + 872);
  if ( v60 )
  {
    if ( !v47 )
    {
      for ( ii = *(_QWORD *)(v2 + 32); v2 + 24 != ii; ii = *(_QWORD *)(ii + 8) )
      {
        v53 = ii - 56;
        if ( !ii )
          v53 = 0;
        sub_B2B950(v53);
      }
      *(_BYTE *)(v2 + 872) = 1;
    }
  }
  else if ( v47 )
  {
    for ( jj = *(_QWORD *)(v2 + 32); v2 + 24 != jj; jj = *(_QWORD *)(jj + 8) )
    {
      v57 = jj - 56;
      if ( !jj )
        v57 = 0;
      sub_B2B9A0(v57);
    }
    *(_BYTE *)(v2 + 872) = 0;
  }
  return v65;
}
