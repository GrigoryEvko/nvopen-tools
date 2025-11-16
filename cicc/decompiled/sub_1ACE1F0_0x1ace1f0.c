// Function: sub_1ACE1F0
// Address: 0x1ace1f0
//
__int64 __fastcall sub_1ACE1F0(__int64 a1, const void *a2, unsigned __int64 a3, unsigned int a4, unsigned int a5)
{
  const void *v5; // r10
  size_t v6; // r9
  __int64 v7; // r15
  __int64 v8; // r14
  _BYTE *v10; // rdx
  unsigned __int64 v11; // rcx
  char *v12; // r14
  size_t v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rdi
  size_t v16; // r14
  void *v17; // r15
  __int64 v18; // rdx
  _BYTE *v19; // rax
  _BYTE *v21; // rdi
  int v22; // r8d
  unsigned __int64 v23; // rax
  _QWORD *v24; // rdi
  int v25; // [rsp+8h] [rbp-198h]
  size_t v26; // [rsp+8h] [rbp-198h]
  _BYTE *v27; // [rsp+18h] [rbp-188h] BYREF
  char v28; // [rsp+34h] [rbp-16Ch] BYREF
  _BYTE v29[11]; // [rsp+35h] [rbp-16Bh] BYREF
  void *src; // [rsp+40h] [rbp-160h] BYREF
  size_t n; // [rsp+48h] [rbp-158h]
  _QWORD v32[2]; // [rsp+50h] [rbp-150h] BYREF
  _BYTE *v33; // [rsp+60h] [rbp-140h] BYREF
  __int64 v34; // [rsp+68h] [rbp-138h]
  _BYTE dest[304]; // [rsp+70h] [rbp-130h] BYREF

  v5 = a2;
  v6 = a3;
  v7 = a5;
  v8 = a4;
  v33 = dest;
  v34 = 0x10000000000LL;
  if ( a3 > 0x100 )
  {
    v26 = a3;
    sub_16CD150((__int64)&v33, dest, a3, 1, a5, a3);
    v6 = v26;
    v5 = a2;
    v21 = &v33[(unsigned int)v34];
  }
  else
  {
    if ( !a3 )
    {
      LODWORD(v34) = 0;
      goto LABEL_4;
    }
    v21 = dest;
  }
  v25 = v6;
  memcpy(v21, v5, v6);
  LODWORD(v6) = v25;
  LODWORD(v34) = v25 + v34;
  a3 = (unsigned int)v34;
  if ( HIDWORD(v34) - (unsigned __int64)(unsigned int)v34 <= 5 )
  {
    sub_16CD150((__int64)&v33, dest, (unsigned int)v34 + 6LL, 1, v22, v25);
    a3 = (unsigned int)v34;
  }
LABEL_4:
  v10 = &v33[a3];
  *(_DWORD *)v10 = 1986817070;
  *((_WORD *)v10 + 2) = 11885;
  LODWORD(v34) = v34 + 6;
  v11 = v7 | (v8 << 32);
  if ( !v11 )
  {
    v28 = 48;
    v12 = &v28;
    src = v32;
LABEL_6:
    v13 = 1;
    LOBYTE(v32[0]) = *v12;
    v14 = v32;
    goto LABEL_7;
  }
  v12 = v29;
  do
  {
    *--v12 = v11 % 0xA + 48;
    v23 = v11;
    v11 /= 0xAu;
  }
  while ( v23 > 9 );
  v13 = v29 - v12;
  src = v32;
  v27 = (_BYTE *)(v29 - v12);
  if ( (unsigned __int64)(v29 - v12) > 0xF )
  {
    src = (void *)sub_22409D0(&src, &v27, 0);
    v24 = src;
    v32[0] = v27;
LABEL_23:
    memcpy(v24, v12, v13);
    v13 = (size_t)v27;
    v14 = src;
    goto LABEL_7;
  }
  if ( v13 == 1 )
    goto LABEL_6;
  if ( v13 )
  {
    v24 = v32;
    goto LABEL_23;
  }
  v14 = v32;
LABEL_7:
  n = v13;
  *((_BYTE *)v14 + v13) = 0;
  v15 = (unsigned int)v34;
  v16 = n;
  v17 = src;
  if ( n > HIDWORD(v34) - (unsigned __int64)(unsigned int)v34 )
  {
    sub_16CD150((__int64)&v33, dest, n + (unsigned int)v34, 1, (int)v32, v6);
    v15 = (unsigned int)v34;
  }
  if ( v16 )
  {
    memcpy(&v33[v15], v17, v16);
    LODWORD(v15) = v34;
  }
  LODWORD(v34) = v15 + v16;
  v18 = (unsigned int)(v15 + v16);
  if ( src != v32 )
  {
    j_j___libc_free_0(src, v32[0] + 1LL);
    v18 = (unsigned int)v34;
  }
  v19 = v33;
  *(_QWORD *)a1 = a1 + 16;
  if ( !v19 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_25;
  }
  sub_1ACE140((__int64 *)a1, v19, (__int64)&v19[v18]);
  v19 = v33;
  if ( v33 != dest )
LABEL_25:
    _libc_free((unsigned __int64)v19);
  return a1;
}
