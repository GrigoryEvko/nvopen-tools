// Function: sub_1B7B450
// Address: 0x1b7b450
//
_QWORD *__fastcall sub_1B7B450(_QWORD *a1)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // r15
  int *v7; // rax
  size_t v8; // rdx
  __int64 v9; // r14
  __int64 i; // r12
  __int64 v11; // r15
  int *v12; // rax
  size_t v13; // rdx
  const void *v14; // r8
  size_t v15; // r15
  _QWORD *v16; // rax
  _BYTE *v17; // rdi
  _BYTE *v18; // rax
  size_t v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // rdi
  _BYTE *src; // [rsp+0h] [rbp-150h]
  void *srca; // [rsp+0h] [rbp-150h]
  _QWORD *v26; // [rsp+8h] [rbp-148h]
  size_t v27; // [rsp+18h] [rbp-138h] BYREF
  _DWORD v28[4]; // [rsp+20h] [rbp-130h] BYREF
  _QWORD *v29; // [rsp+30h] [rbp-120h] BYREF
  _BYTE *v30; // [rsp+38h] [rbp-118h]
  _QWORD v31[2]; // [rsp+40h] [rbp-110h] BYREF
  _BYTE *v32; // [rsp+50h] [rbp-100h] BYREF
  size_t n; // [rsp+58h] [rbp-F8h]
  _BYTE v34[32]; // [rsp+60h] [rbp-F0h] BYREF
  int v35[52]; // [rsp+80h] [rbp-D0h] BYREF

  v26 = a1 + 1;
  src = (_BYTE *)a1[2];
  if ( src )
    return v26;
  sub_16C1840(v35);
  v3 = *a1;
  v4 = *(_QWORD *)(*a1 + 32LL);
  v5 = *a1 + 24LL;
  if ( v4 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = 0;
        if ( v4 )
          v6 = v4 - 56;
        if ( !sub_15E4F60(v6) && (*(_BYTE *)(v6 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v6 + 23) & 0x20) != 0 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v5 == v4 )
          goto LABEL_12;
      }
      v7 = (int *)sub_1649960(v6);
      sub_16C1A90(v35, v7, v8);
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v5 != v4 );
LABEL_12:
    v3 = *a1;
  }
  v9 = *(_QWORD *)(v3 + 16);
  for ( i = v3 + 8; i != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    while ( 1 )
    {
      v11 = 0;
      if ( v9 )
        v11 = v9 - 56;
      if ( !sub_15E4F60(v11) && (*(_BYTE *)(v11 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v11 + 23) & 0x20) != 0 )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( i == v9 )
        goto LABEL_22;
    }
    v12 = (int *)sub_1649960(v11);
    sub_16C1A90(v35, v12, v13);
  }
LABEL_22:
  sub_16C1AA0(v35, v28);
  v32 = v34;
  n = 0x2000000000LL;
  sub_16C1D70((char *)v28, (__int64)&v32);
  v14 = v32;
  if ( !v32 )
  {
    LOBYTE(v31[0]) = 0;
    v17 = (_BYTE *)a1[1];
    v29 = v31;
LABEL_35:
    a1[2] = src;
    src[(_QWORD)v17] = 0;
    v18 = v29;
    goto LABEL_30;
  }
  v15 = (unsigned int)n;
  v29 = v31;
  v27 = (unsigned int)n;
  if ( (unsigned int)n > 0xFuLL )
  {
    srca = v32;
    v22 = sub_22409D0(&v29, &v27, 0);
    v14 = srca;
    v29 = (_QWORD *)v22;
    v23 = (_QWORD *)v22;
    v31[0] = v27;
LABEL_37:
    memcpy(v23, v14, v15);
    v15 = v27;
    v16 = v29;
    goto LABEL_26;
  }
  if ( (unsigned int)n == 1 )
  {
    LOBYTE(v31[0]) = *v32;
    v16 = v31;
    goto LABEL_26;
  }
  if ( (_DWORD)n )
  {
    v23 = v31;
    goto LABEL_37;
  }
  v16 = v31;
LABEL_26:
  v30 = (_BYTE *)v15;
  *((_BYTE *)v16 + v15) = 0;
  v17 = (_BYTE *)a1[1];
  v18 = v17;
  if ( v29 == v31 )
  {
    src = v30;
    if ( v30 )
    {
      if ( v30 == (_BYTE *)1 )
        *v17 = v31[0];
      else
        memcpy(v17, v31, (size_t)v30);
      v17 = (_BYTE *)a1[1];
      src = v30;
    }
    goto LABEL_35;
  }
  v19 = (size_t)v30;
  v20 = v31[0];
  if ( v18 == (_BYTE *)(a1 + 3) )
  {
    a1[1] = v29;
    a1[2] = v19;
    a1[3] = v20;
    goto LABEL_39;
  }
  v21 = a1[3];
  a1[1] = v29;
  a1[2] = v19;
  a1[3] = v20;
  if ( !v18 )
  {
LABEL_39:
    v29 = v31;
    v18 = v31;
    goto LABEL_30;
  }
  v29 = v18;
  v31[0] = v21;
LABEL_30:
  v30 = 0;
  *v18 = 0;
  if ( v29 != v31 )
    j_j___libc_free_0(v29, v31[0] + 1LL);
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  return v26;
}
