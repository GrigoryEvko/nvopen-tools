// Function: sub_25C4870
// Address: 0x25c4870
//
void __fastcall sub_25C4870(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v5; // rbx
  bool v6; // zf
  __int64 v7; // rsi
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // r12
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // r15
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  char v21; // al
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // r8
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v29; // r8
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r14
  unsigned __int64 i; // rbx
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // r15
  __int64 v39; // r8
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  __int64 v42; // r15
  __int64 v43; // r8
  unsigned __int64 v44; // r12
  unsigned __int64 v45; // rdi
  __int64 *v46; // [rsp+0h] [rbp-120h]
  __int64 v47; // [rsp+0h] [rbp-120h]
  __int64 v48; // [rsp+8h] [rbp-118h]
  __int64 v49; // [rsp+10h] [rbp-110h]
  unsigned __int64 v50; // [rsp+10h] [rbp-110h]
  unsigned __int64 v51; // [rsp+18h] [rbp-108h]
  __int64 v52; // [rsp+30h] [rbp-F0h]
  char v53; // [rsp+38h] [rbp-E8h]
  _BYTE *v54; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+48h] [rbp-D8h]
  _BYTE v56[64]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+90h] [rbp-90h] BYREF
  __int64 *v58; // [rsp+98h] [rbp-88h] BYREF
  __int64 v59; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-78h] BYREF
  _BYTE v61[112]; // [rsp+B0h] [rbp-70h] BYREF

  v3 = a2 - a1;
  v51 = a2;
  v49 = a3;
  if ( (__int64)(a2 - a1) <= 1536 )
    return;
  if ( !a3 )
  {
    v13 = a2;
    goto LABEL_71;
  }
  v48 = a1 + 96;
  v46 = (__int64 *)(a1 + 8);
  while ( 2 )
  {
    --v49;
    v5 = (__int64 *)(a1
                   + 32
                   * (((0xAAAAAAAAAAAAAAABLL * (v3 >> 5)) & 0xFFFFFFFFFFFFFFFELL)
                    + ((__int64)(0xAAAAAAAAAAAAAAABLL * (v3 >> 5)) >> 1)));
    v6 = !sub_B445A0(*(_QWORD *)(a1 + 96), *v5);
    v7 = *(_QWORD *)(v51 - 96);
    if ( v6 )
    {
      if ( sub_B445A0(*(_QWORD *)(a1 + 96), v7) )
      {
        v20 = *(_QWORD *)(a1 + 96);
        *(_QWORD *)(a1 + 96) = *(_QWORD *)a1;
        v21 = *(_BYTE *)(a1 + 8);
        *(_QWORD *)a1 = v20;
        LODWORD(v20) = *(_DWORD *)(a1 + 24);
        LOBYTE(v57) = v21;
        v58 = &v60;
        v59 = 0x200000000LL;
        if ( (_DWORD)v20 )
          sub_25C2C90((__int64)&v58, (__int64 *)(a1 + 16));
        v10 = a1 + 112;
        *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 104);
        sub_25C2C90(a1 + 16, (__int64 *)(a1 + 112));
        *(_BYTE *)(a1 + 104) = v57;
        goto LABEL_9;
      }
      v6 = !sub_B445A0(*v5, *(_QWORD *)(v51 - 96));
      v32 = *(_QWORD *)a1;
      if ( v6 )
      {
        *(_QWORD *)a1 = *v5;
        *v5 = v32;
        sub_25C3260(v46, v5 + 1);
        goto LABEL_10;
      }
LABEL_65:
      *(_QWORD *)a1 = *(_QWORD *)(v51 - 96);
      *(_QWORD *)(v51 - 96) = v32;
      sub_25C3260(v46, (__int64 *)(v51 - 88));
      goto LABEL_10;
    }
    if ( !sub_B445A0(*v5, v7) )
    {
      v6 = !sub_B445A0(*(_QWORD *)(a1 + 96), *(_QWORD *)(v51 - 96));
      v32 = *(_QWORD *)a1;
      if ( v6 )
      {
        v33 = *(_QWORD *)(a1 + 96);
        *(_QWORD *)(a1 + 96) = v32;
        *(_QWORD *)a1 = v33;
        sub_25C3260(v46, (__int64 *)(a1 + 104));
        goto LABEL_10;
      }
      goto LABEL_65;
    }
    v8 = *(_QWORD *)a1;
    *(_QWORD *)a1 = *v5;
    *v5 = v8;
    v9 = *(_DWORD *)(a1 + 24);
    LOBYTE(v57) = *(_BYTE *)(a1 + 8);
    v58 = &v60;
    v59 = 0x200000000LL;
    if ( v9 )
      sub_25C2C90((__int64)&v58, (__int64 *)(a1 + 16));
    v10 = (__int64)(v5 + 2);
    *(_BYTE *)(a1 + 8) = *((_BYTE *)v5 + 8);
    sub_25C2C90(a1 + 16, v5 + 2);
    *((_BYTE *)v5 + 8) = v57;
LABEL_9:
    sub_25C2C90(v10, (__int64 *)&v58);
    sub_25C0430((__int64)&v58);
LABEL_10:
    v11 = v48;
    v12 = v51;
    while ( 1 )
    {
      v13 = v11;
      if ( sub_B445A0(*(_QWORD *)v11, *(_QWORD *)a1) )
        goto LABEL_11;
      do
      {
        v14 = *(_QWORD *)(v12 - 96);
        v12 -= 96LL;
      }
      while ( sub_B445A0(*(_QWORD *)a1, v14) );
      if ( v11 >= v12 )
        break;
      v15 = *(_QWORD *)v11;
      v16 = v11 + 16;
      *(_QWORD *)v11 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v15;
      LOBYTE(v57) = *(_BYTE *)(v11 + 8);
      v58 = &v60;
      v59 = 0x200000000LL;
      if ( *(_DWORD *)(v11 + 24) )
      {
        sub_25C2C90((__int64)&v58, (__int64 *)(v11 + 16));
        v16 = v11 + 16;
      }
      *(_BYTE *)(v11 + 8) = *(_BYTE *)(v12 + 8);
      sub_25C2C90(v16, (__int64 *)(v12 + 16));
      *(_BYTE *)(v12 + 8) = v57;
      sub_25C2C90(v12 + 16, (__int64 *)&v58);
      v17 = (__int64)v58;
      v18 = (unsigned __int64)&v58[4 * (unsigned int)v59];
      if ( v58 != (__int64 *)v18 )
      {
        do
        {
          v18 -= 32LL;
          if ( *(_DWORD *)(v18 + 24) > 0x40u )
          {
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
          if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
            j_j___libc_free_0_0(*(_QWORD *)v18);
        }
        while ( v17 != v18 );
        v18 = (unsigned __int64)v58;
      }
      if ( (__int64 *)v18 != &v60 )
        _libc_free(v18);
LABEL_11:
      v11 += 96LL;
    }
    sub_25C4870(v11, v51, v49);
    v3 = v11 - a1;
    if ( (__int64)(v11 - a1) > 1536 )
    {
      if ( v49 )
      {
        v51 = v11;
        continue;
      }
LABEL_71:
      v50 = v13;
      v47 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 5);
      v34 = (v47 - 2) >> 1;
      for ( i = a1 + 32 * (v34 + ((v47 - 2) & 0xFFFFFFFFFFFFFFFELL)) + 16; ; i -= 96LL )
      {
        v36 = *(_QWORD *)(i - 16);
        v52 = v36;
        v37 = *(_BYTE *)(i - 8);
        v54 = v56;
        v53 = v37;
        v55 = 0x200000000LL;
        if ( *(_DWORD *)(i + 8) )
        {
          sub_25C2C90((__int64)&v54, (__int64 *)i);
          v57 = v52;
          LOBYTE(v58) = v53;
          v59 = (__int64)v61;
          v60 = 0x200000000LL;
          if ( (_DWORD)v55 )
            sub_25C2C90((__int64)&v59, (__int64 *)&v54);
        }
        else
        {
          LOBYTE(v58) = v37;
          v57 = v36;
          v59 = (__int64)v61;
          v60 = 0x200000000LL;
        }
        sub_25C3660(a1, v34, v47, (__int64)&v57);
        v38 = v59;
        v39 = 32LL * (unsigned int)v60;
        v40 = v59 + v39;
        if ( v59 != v59 + v39 )
        {
          do
          {
            v40 -= 32LL;
            if ( *(_DWORD *)(v40 + 24) > 0x40u )
            {
              v41 = *(_QWORD *)(v40 + 16);
              if ( v41 )
                j_j___libc_free_0_0(v41);
            }
            if ( *(_DWORD *)(v40 + 8) > 0x40u && *(_QWORD *)v40 )
              j_j___libc_free_0_0(*(_QWORD *)v40);
          }
          while ( v38 != v40 );
          v40 = v59;
        }
        if ( (_BYTE *)v40 != v61 )
          _libc_free(v40);
        if ( !v34 )
          break;
        v42 = (__int64)v54;
        --v34;
        v43 = 32LL * (unsigned int)v55;
        v44 = (unsigned __int64)&v54[v43];
        if ( v54 != &v54[v43] )
        {
          do
          {
            v44 -= 32LL;
            if ( *(_DWORD *)(v44 + 24) > 0x40u )
            {
              v45 = *(_QWORD *)(v44 + 16);
              if ( v45 )
                j_j___libc_free_0_0(v45);
            }
            if ( *(_DWORD *)(v44 + 8) > 0x40u && *(_QWORD *)v44 )
              j_j___libc_free_0_0(*(_QWORD *)v44);
          }
          while ( v42 != v44 );
          v44 = (unsigned __int64)v54;
        }
        if ( (_BYTE *)v44 != v56 )
          _libc_free(v44);
      }
      v22 = v50 - 80;
      sub_25C0430((__int64)&v54);
      do
      {
        v52 = *(_QWORD *)(v22 - 16);
        v53 = *(_BYTE *)(v22 - 8);
        v54 = v56;
        v55 = 0x200000000LL;
        if ( *(_DWORD *)(v22 + 8) )
          sub_25C2C90((__int64)&v54, (__int64 *)v22);
        *(_QWORD *)(v22 - 16) = *(_QWORD *)a1;
        *(_BYTE *)(v22 - 8) = *(_BYTE *)(a1 + 8);
        sub_25C2C90(v22, (__int64 *)(a1 + 16));
        v57 = v52;
        LOBYTE(v58) = v53;
        v59 = (__int64)v61;
        v60 = 0x200000000LL;
        if ( (_DWORD)v55 )
          sub_25C2C90((__int64)&v59, (__int64 *)&v54);
        v23 = v22 - 16 - a1;
        sub_25C3660(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v23 >> 5), (__int64)&v57);
        v24 = v59;
        v25 = 32LL * (unsigned int)v60;
        v26 = v59 + v25;
        if ( v59 != v59 + v25 )
        {
          do
          {
            v26 -= 32LL;
            if ( *(_DWORD *)(v26 + 24) > 0x40u )
            {
              v27 = *(_QWORD *)(v26 + 16);
              if ( v27 )
                j_j___libc_free_0_0(v27);
            }
            if ( *(_DWORD *)(v26 + 8) > 0x40u && *(_QWORD *)v26 )
              j_j___libc_free_0_0(*(_QWORD *)v26);
          }
          while ( v24 != v26 );
          v26 = v59;
        }
        if ( (_BYTE *)v26 != v61 )
          _libc_free(v26);
        v28 = (__int64)v54;
        v29 = 32LL * (unsigned int)v55;
        v30 = (unsigned __int64)&v54[v29];
        if ( v54 != &v54[v29] )
        {
          do
          {
            v30 -= 32LL;
            if ( *(_DWORD *)(v30 + 24) > 0x40u )
            {
              v31 = *(_QWORD *)(v30 + 16);
              if ( v31 )
                j_j___libc_free_0_0(v31);
            }
            if ( *(_DWORD *)(v30 + 8) > 0x40u && *(_QWORD *)v30 )
              j_j___libc_free_0_0(*(_QWORD *)v30);
          }
          while ( v28 != v30 );
          v30 = (unsigned __int64)v54;
        }
        if ( (_BYTE *)v30 != v56 )
          _libc_free(v30);
        v22 -= 96;
      }
      while ( v23 > 96 );
    }
    break;
  }
}
