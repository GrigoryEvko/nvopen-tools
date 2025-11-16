// Function: sub_F6F7A0
// Address: 0xf6f7a0
//
__int64 __fastcall sub_F6F7A0(__int64 *a1, void **p_s, int a3, int a4, __int64 a5, __int64 a6)
{
  unsigned int *v7; // rax
  unsigned __int64 v8; // r13
  char *v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rbx
  char *v12; // rdx
  char *v13; // rdi
  unsigned int v14; // r15d
  unsigned int v15; // eax
  __int64 v16; // rdx
  char *v17; // rdx
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r15
  __int64 v23; // r13
  char *v25; // rax
  char *v26; // rdx
  __int64 v27; // rdx
  int i; // ecx
  size_t v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rdx
  unsigned int v32; // esi
  _QWORD *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  int v38; // [rsp+14h] [rbp-12Ch]
  unsigned int v39; // [rsp+14h] [rbp-12Ch]
  int v40; // [rsp+18h] [rbp-128h] BYREF
  int v41; // [rsp+1Ch] [rbp-124h] BYREF
  void **v42; // [rsp+28h] [rbp-118h] BYREF
  _QWORD v43[4]; // [rsp+30h] [rbp-110h] BYREF
  char v44[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v45; // [rsp+70h] [rbp-D0h]
  void *s; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+88h] [rbp-B8h]
  _DWORD v48[4]; // [rsp+90h] [rbp-B0h] BYREF
  __int16 v49; // [rsp+A0h] [rbp-A0h]

  v7 = (unsigned int *)p_s[1];
  v41 = a3;
  v40 = a5;
  v8 = v7[8];
  v43[0] = a1;
  v43[1] = &v41;
  v43[2] = &v40;
  v9 = (char *)v48;
  v10 = v8;
  v42 = p_s;
  s = v48;
  if ( a4 == 1 )
  {
    v47 = 0x2000000000LL;
    if ( !v8 )
      goto LABEL_21;
    if ( v8 > 0x20 )
    {
      p_s = (void **)v48;
      sub_C8D5F0((__int64)&s, v48, v8, 4u, a5, a6);
      v13 = (char *)s;
      v26 = (char *)s + 4 * v8;
      v25 = (char *)s + 4 * (unsigned int)v47;
      if ( v26 == v25 )
      {
        LODWORD(v47) = v8;
LABEL_30:
        v27 = (unsigned int)v8;
        for ( i = 1; ; i = v39 )
        {
          v29 = 4 * v27;
          if ( v29 )
          {
            v38 = i;
            memset(v13, 255, v29);
            v13 = (char *)s;
            i = v38;
          }
          v30 = 0;
          while ( 1 )
          {
            v31 = v30;
            v32 = v30 + i;
            v30 += 2 * i;
            *(_DWORD *)&v13[4 * v31] = v32;
            if ( (unsigned int)v8 <= v30 )
              break;
            v13 = (char *)s;
          }
          p_s = &s;
          v39 = 2 * i;
          sub_F6F410((__int64)v43, (__int64)&s, (__int64 *)&v42);
          v13 = (char *)s;
          if ( v39 >= (unsigned int)v8 )
            break;
          v27 = (unsigned int)v47;
        }
        goto LABEL_19;
      }
    }
    else
    {
      v25 = (char *)v48;
      v26 = (char *)&v48[v8];
    }
    do
    {
      if ( v25 )
        *(_DWORD *)v25 = 0;
      v25 += 4;
    }
    while ( v26 != v25 );
    LODWORD(v47) = v8;
    v13 = (char *)s;
    if ( (_DWORD)v8 == 1 )
      goto LABEL_19;
    goto LABEL_30;
  }
  v47 = 0x2000000000LL;
  if ( !v8 )
  {
    v13 = (char *)v48;
    v10 = 0;
    goto LABEL_8;
  }
  v11 = 4 * v8;
  v12 = (char *)&v48[v8];
  if ( v8 > 0x20 )
  {
    p_s = (void **)v48;
    sub_C8D5F0((__int64)&s, v48, v8, 4u, a5, a6);
    v13 = (char *)s;
    v12 = (char *)s + v11;
    v9 = (char *)s + 4 * (unsigned int)v47;
    if ( v9 == (char *)s + v11 )
    {
      LODWORD(v47) = v8;
LABEL_8:
      v14 = v10 >> 1;
      if ( !v14 )
        goto LABEL_16;
      while ( 1 )
      {
        v15 = v14;
        v16 = 0;
        while ( 1 )
        {
          *(_DWORD *)&v13[v16] = v15++;
          v16 += 4;
          if ( 2 * v14 == v15 )
            break;
          v13 = (char *)s;
        }
        v17 = (char *)s + 4 * (unsigned int)v47;
        v13 = (char *)s + 4 * v14;
        if ( v13 != v17 )
LABEL_13:
          memset(v13, 255, v17 - v13);
        p_s = &s;
        sub_F6F410((__int64)v43, (__int64)&s, (__int64 *)&v42);
        if ( v14 == 1 )
          break;
        while ( 1 )
        {
          v14 >>= 1;
          v13 = (char *)s;
          if ( v14 )
            break;
LABEL_16:
          v17 = &v13[4 * (unsigned int)v47];
          if ( v17 != v13 )
            goto LABEL_13;
          sub_F6F410((__int64)v43, (__int64)&s, (__int64 *)&v42);
        }
      }
      v13 = (char *)s;
      goto LABEL_19;
    }
  }
  do
  {
    if ( v9 )
      *(_DWORD *)v9 = 0;
    v9 += 4;
  }
  while ( v9 != v12 );
  LODWORD(v47) = v8;
  v13 = (char *)s;
  if ( (_DWORD)v8 != 1 )
    goto LABEL_8;
LABEL_19:
  if ( v13 != (char *)v48 )
    _libc_free(v13, p_s);
LABEL_21:
  v18 = (_QWORD *)a1[9];
  v45 = 257;
  v19 = sub_BCB2D0(v18);
  v20 = sub_ACD640(v19, 0, 0);
  v21 = (__int64)v42;
  v22 = v20;
  v23 = (*(__int64 (__fastcall **)(__int64, void **, __int64))(*(_QWORD *)a1[10] + 96LL))(a1[10], v42, v20);
  if ( !v23 )
  {
    v49 = 257;
    v33 = sub_BD2C40(72, 2u);
    v23 = (__int64)v33;
    if ( v33 )
      sub_B4DE80((__int64)v33, v21, v22, (__int64)&s, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v23,
      v44,
      a1[7],
      a1[8]);
    v34 = *a1;
    v35 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v35 )
    {
      do
      {
        v36 = *(_QWORD *)(v34 + 8);
        v37 = *(_DWORD *)v34;
        v34 += 16;
        sub_B99FD0(v23, v37, v36);
      }
      while ( v35 != v34 );
    }
  }
  return v23;
}
