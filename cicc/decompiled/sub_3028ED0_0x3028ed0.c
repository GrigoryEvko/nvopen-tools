// Function: sub_3028ED0
// Address: 0x3028ed0
//
__int64 __fastcall sub_3028ED0(__int64 a1, unsigned int *a2, size_t a3)
{
  size_t v3; // rcx
  __int64 v4; // r13
  _BYTE *v7; // rdi
  unsigned int *v8; // rax
  size_t v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 result; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rsi
  _BYTE *v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rcx
  __int64 v22; // rdx
  size_t v23; // rdx
  unsigned int *v24; // [rsp+0h] [rbp-90h]
  size_t v26; // [rsp+8h] [rbp-88h]
  _BYTE *v27; // [rsp+8h] [rbp-88h]
  _QWORD v28[2]; // [rsp+10h] [rbp-80h] BYREF
  char v29; // [rsp+20h] [rbp-70h]
  unsigned int *v30; // [rsp+30h] [rbp-60h] BYREF
  size_t n; // [rsp+38h] [rbp-58h]
  _QWORD src[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v33; // [rsp+50h] [rbp-40h]

  v3 = a3;
  v4 = a1 + 8;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  if ( !a2 )
  {
    LOBYTE(src[0]) = 0;
    v23 = 0;
    v30 = (unsigned int *)src;
    v7 = (_BYTE *)(a1 + 80);
LABEL_28:
    *(_QWORD *)(a1 + 72) = v23;
    v7[v23] = 0;
    v8 = v30;
    goto LABEL_6;
  }
  v24 = (unsigned int *)(a1 + 80);
  v30 = (unsigned int *)src;
  sub_3020610((__int64 *)&v30, a2, (__int64)a2 + a3);
  v7 = *(_BYTE **)(a1 + 64);
  v3 = a3;
  v8 = *(unsigned int **)(a1 + 64);
  if ( v30 == (unsigned int *)src )
  {
    v23 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *v7 = src[0];
        v23 = n;
        v7 = *(_BYTE **)(a1 + 64);
      }
      else
      {
        memcpy(v7, src, n);
        v23 = n;
        v7 = *(_BYTE **)(a1 + 64);
        v3 = a3;
      }
    }
    goto LABEL_28;
  }
  v9 = n;
  v10 = src[0];
  if ( v24 == v8 )
  {
    *(_QWORD *)(a1 + 64) = v30;
    *(_QWORD *)(a1 + 72) = v9;
    *(_QWORD *)(a1 + 80) = v10;
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 64) = v30;
    *(_QWORD *)(a1 + 72) = v9;
    *(_QWORD *)(a1 + 80) = v10;
    if ( v8 )
    {
      v30 = v8;
      src[0] = v11;
      goto LABEL_6;
    }
  }
  v30 = (unsigned int *)src;
  v8 = (unsigned int *)src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v30 != (unsigned int *)src )
  {
    v26 = v3;
    j_j___libc_free_0((unsigned __int64)v30);
    v3 = v26;
  }
  v30 = a2;
  v33 = 261;
  n = v3;
  result = sub_C7EA90((__int64)v28, (__int64 *)&v30, 0, 1u, 0, 0);
  if ( (v29 & 1) == 0 )
  {
    v14 = v28[0];
    v13 = *(_QWORD *)(a1 + 56);
    v28[0] = 0;
    *(_QWORD *)(a1 + 56) = v14;
    if ( !v13 )
      goto LABEL_18;
    goto LABEL_16;
  }
  v13 = *(_QWORD *)(a1 + 56);
  if ( LODWORD(v28[0]) )
  {
    *(_QWORD *)(a1 + 56) = 0;
    if ( v13 )
    {
      result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
      *(_BYTE *)(a1 + 48) = 1;
      if ( (v29 & 1) == 0 )
      {
        if ( v28[0] )
          return (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v28[0] + 8LL))(v28[0]);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 48) = 1;
    }
    return result;
  }
  v14 = v28[0];
  v28[0] = 0;
  *(_QWORD *)(a1 + 56) = v14;
  if ( v13 )
  {
LABEL_16:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    if ( (v29 & 1) == 0 && v28[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v28[0] + 8LL))(v28[0]);
    v14 = *(_QWORD *)(a1 + 56);
  }
LABEL_18:
  *(_BYTE *)(a1 + 48) = *(_QWORD *)(v14 + 16) == *(_QWORD *)(v14 + 8);
  v15 = sub_3028300(a1, *(_BYTE **)(v14 + 8));
  LODWORD(v28[0]) = 1;
  v16 = v4;
  v17 = v15;
  v18 = *(_QWORD *)(a1 + 16);
  v20 = v19;
  if ( !v18 )
    goto LABEL_25;
  do
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(v18 + 16);
      v22 = *(_QWORD *)(v18 + 24);
      if ( *(_DWORD *)(v18 + 32) <= 1u )
        break;
      v18 = *(_QWORD *)(v18 + 24);
      if ( !v22 )
        goto LABEL_23;
    }
    v16 = v18;
    v18 = *(_QWORD *)(v18 + 16);
  }
  while ( v21 );
LABEL_23:
  if ( v4 == v16 || (result = *(unsigned int *)(v16 + 32), !(_DWORD)result) )
  {
LABEL_25:
    v27 = v17;
    v30 = (unsigned int *)v28;
    result = sub_3028E10((_QWORD *)a1, v16, &v30);
    v17 = v27;
    v16 = result;
  }
  *(_QWORD *)(v16 + 40) = v17;
  *(_QWORD *)(v16 + 48) = v20;
  *(_BYTE *)(v16 + 56) = 0;
  return result;
}
