// Function: sub_1CCDC20
// Address: 0x1ccdc20
//
__int64 __fastcall sub_1CCDC20(__int64 a1)
{
  __int64 **v1; // rbx
  __int64 *v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  int v5; // r10d
  __int64 *v6; // r9
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // rsi
  int v13; // edx
  int v14; // r10d
  __int64 *v15; // r8
  __int64 v16; // rdx
  unsigned int v17; // eax
  unsigned int i; // r15d
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 **v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 *v25; // rbx
  __int64 *v26; // r13
  __int64 v27; // rdi
  int v29; // r8d
  unsigned int v30; // r15d
  __int64 *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v35; // [rsp+18h] [rbp-98h]
  __int64 v36; // [rsp+20h] [rbp-90h]
  __int64 v37; // [rsp+28h] [rbp-88h]
  __int64 v38; // [rsp+30h] [rbp-80h] BYREF
  __int64 v39; // [rsp+38h] [rbp-78h]
  __int64 *v40; // [rsp+40h] [rbp-70h]
  __int64 *v41; // [rsp+48h] [rbp-68h]
  _QWORD *v42; // [rsp+50h] [rbp-60h]
  unsigned __int64 v43; // [rsp+58h] [rbp-58h]
  __int64 v44; // [rsp+60h] [rbp-50h]
  __int64 *v45; // [rsp+68h] [rbp-48h]
  _QWORD *v46; // [rsp+70h] [rbp-40h]
  __int64 **v47; // [rsp+78h] [rbp-38h]

  v34 = 0;
  v33 = sub_1CCAE90(a1, 1);
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v44 = 0;
  v39 = 8;
  v38 = sub_22077B0(64);
  v1 = (__int64 **)(v38 + 24);
  v43 = v38 + 24;
  v2 = (__int64 *)sub_22077B0(512);
  *(_QWORD *)(v38 + 24) = v2;
  v41 = v2;
  v42 = v2 + 64;
  v47 = v1;
  v45 = v2;
  v46 = v2 + 64;
  v40 = v2;
  if ( v2 )
  {
    v3 = (__int64)(v2 + 1);
    *v2 = v33;
    v44 = (__int64)(v2 + 1);
  }
  else
  {
    v44 = 8;
    v3 = 8;
  }
  v4 = 0;
  while ( 1 )
  {
    if ( v2 == (__int64 *)v3 )
    {
      v10 = (*(v1 - 1))[63];
      j_j___libc_free_0(v2, 512);
      v16 = (__int64)(*--v47 + 64);
      v45 = *v47;
      v46 = (_QWORD *)v16;
      v44 = (__int64)(v45 + 63);
    }
    else
    {
      v10 = *(_QWORD *)(v3 - 8);
      v44 = v3 - 8;
    }
    if ( !(_DWORD)v37 )
    {
      ++v34;
LABEL_11:
      sub_13B3B90((__int64)&v34, 2 * v37);
      if ( !(_DWORD)v37 )
        goto LABEL_78;
      v11 = (v37 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v6 = (__int64 *)(v35 + 8LL * v11);
      v12 = *v6;
      v13 = v36 + 1;
      if ( v10 != *v6 )
      {
        v14 = 1;
        v15 = 0;
        while ( v12 != -8 )
        {
          if ( !v15 && v12 == -16 )
            v15 = v6;
          v11 = (v37 - 1) & (v14 + v11);
          v6 = (__int64 *)(v35 + 8LL * v11);
          v12 = *v6;
          if ( v10 == *v6 )
            goto LABEL_27;
          ++v14;
        }
        if ( v15 )
          v6 = v15;
      }
      goto LABEL_27;
    }
    v5 = 1;
    v6 = 0;
    v7 = (v37 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v8 = (__int64 *)(v35 + 8LL * v7);
    v9 = *v8;
    if ( v10 == *v8 )
      goto LABEL_5;
    while ( v9 != -8 )
    {
      if ( v9 != -16 || v6 )
        v8 = v6;
      v7 = (v37 - 1) & (v5 + v7);
      v9 = *(_QWORD *)(v35 + 8LL * v7);
      if ( v10 == v9 )
        goto LABEL_5;
      ++v5;
      v6 = v8;
      v8 = (__int64 *)(v35 + 8LL * v7);
    }
    if ( !v6 )
      v6 = v8;
    ++v34;
    v13 = v36 + 1;
    if ( 4 * ((int)v36 + 1) >= (unsigned int)(3 * v37) )
      goto LABEL_11;
    if ( (int)v37 - HIDWORD(v36) - v13 <= (unsigned int)v37 >> 3 )
    {
      sub_13B3B90((__int64)&v34, v37);
      if ( !(_DWORD)v37 )
      {
LABEL_78:
        LODWORD(v36) = v36 + 1;
        BUG();
      }
      v29 = 1;
      v30 = (v37 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v6 = (__int64 *)(v35 + 8LL * v30);
      v13 = v36 + 1;
      v31 = 0;
      v32 = *v6;
      if ( v10 != *v6 )
      {
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v31 )
            v31 = v6;
          v30 = (v37 - 1) & (v29 + v30);
          v6 = (__int64 *)(v35 + 8LL * v30);
          v32 = *v6;
          if ( v10 == *v6 )
            goto LABEL_27;
          ++v29;
        }
        if ( v31 )
          v6 = v31;
      }
    }
LABEL_27:
    LODWORD(v36) = v13;
    if ( *v6 != -8 )
      --HIDWORD(v36);
    *v6 = v10;
    if ( *(_BYTE *)(v10 + 16) != 77 )
      break;
    v17 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
    if ( v17 )
    {
      for ( i = 0; i != v17; ++i )
      {
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
          v19 = *(_QWORD *)(v10 - 8);
        else
          v19 = v10 - 24LL * v17;
        v20 = sub_1CCAE90(*(_QWORD *)(v19 + 24LL * i), 1);
        v21 = v44;
        if ( (_QWORD *)v44 == v46 - 1 )
        {
          v22 = v47;
          if ( ((v44 - (__int64)v45) >> 3) + ((((__int64)((__int64)v47 - v43) >> 3) - 1) << 6) + v42 - v40 == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
          if ( (unsigned __int64)(v39 - (((__int64)v47 - v38) >> 3)) <= 1 )
          {
            sub_1CCDAA0(&v38, 1u, 0);
            v22 = v47;
          }
          v22[1] = (__int64 *)sub_22077B0(512);
          if ( v44 )
            *(_QWORD *)v44 = v20;
          v23 = (__int64)(*++v47 + 64);
          v45 = *v47;
          v46 = (_QWORD *)v23;
          v44 = (__int64)v45;
        }
        else
        {
          if ( v44 )
          {
            *(_QWORD *)v44 = v20;
            v21 = v44;
          }
          v44 = v21 + 8;
        }
        v17 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
      }
    }
LABEL_5:
    v3 = v44;
    if ( v40 == (__int64 *)v44 )
      goto LABEL_51;
LABEL_6:
    v2 = v45;
    v1 = v47;
  }
  if ( v4 )
  {
    if ( v10 != v4 )
    {
      v4 = v33;
      goto LABEL_53;
    }
    goto LABEL_5;
  }
  v3 = v44;
  v4 = v10;
  if ( v40 != (__int64 *)v44 )
    goto LABEL_6;
LABEL_51:
  if ( !v4 )
    v4 = v33;
LABEL_53:
  v24 = v38;
  if ( v38 )
  {
    v25 = (__int64 *)v43;
    v26 = (__int64 *)(v47 + 1);
    if ( (unsigned __int64)(v47 + 1) > v43 )
    {
      do
      {
        v27 = *v25++;
        j_j___libc_free_0(v27, 512);
      }
      while ( v26 > v25 );
      v24 = v38;
    }
    j_j___libc_free_0(v24, 8 * v39);
  }
  j___libc_free_0(v35);
  return v4;
}
