// Function: sub_2A43790
// Address: 0x2a43790
//
_QWORD *__fastcall sub_2A43790(_QWORD *a1)
{
  _QWORD *v1; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r15
  int *v8; // rax
  size_t v9; // rdx
  __int64 v10; // rbx
  __int64 i; // r13
  __int64 v12; // r15
  int *v13; // rax
  size_t v14; // rdx
  void *v15; // r15
  unsigned __int64 v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  size_t v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _QWORD *v22; // rdi
  size_t v23; // rdx
  size_t v24; // [rsp+8h] [rbp-148h] BYREF
  _DWORD v25[4]; // [rsp+10h] [rbp-140h] BYREF
  _QWORD *v26; // [rsp+20h] [rbp-130h] BYREF
  size_t v27; // [rsp+28h] [rbp-128h]
  _QWORD v28[2]; // [rsp+30h] [rbp-120h] BYREF
  void *src; // [rsp+40h] [rbp-110h] BYREF
  size_t n; // [rsp+48h] [rbp-108h]
  __int64 v31; // [rsp+50h] [rbp-100h]
  _BYTE v32[40]; // [rsp+58h] [rbp-F8h] BYREF
  int v33[52]; // [rsp+80h] [rbp-D0h] BYREF

  v1 = a1 + 1;
  if ( a1[2] )
    return v1;
  sub_C7D030(v33);
  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 32LL);
  v6 = *a1 + 24LL;
  if ( v5 != v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = 0;
        if ( v5 )
          v7 = v5 - 56;
        if ( !sub_B2FC80(v7) && (*(_BYTE *)(v7 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v7 + 7) & 0x10) != 0 )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v6 == v5 )
          goto LABEL_12;
      }
      v8 = (int *)sub_BD5D20(v7);
      sub_C7D280(v33, v8, v9);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v6 != v5 );
LABEL_12:
    v4 = *a1;
  }
  v10 = *(_QWORD *)(v4 + 16);
  for ( i = v4 + 8; i != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    while ( 1 )
    {
      v12 = 0;
      if ( v10 )
        v12 = v10 - 56;
      if ( !sub_B2FC80(v12) && (*(_BYTE *)(v12 + 32) & 0xFu) - 7 > 1 && (*(_BYTE *)(v12 + 7) & 0x10) != 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( i == v10 )
        goto LABEL_22;
    }
    v13 = (int *)sub_BD5D20(v12);
    sub_C7D280(v33, v13, v14);
  }
LABEL_22:
  sub_C7D290(v33, v25);
  src = v32;
  n = 0;
  v31 = 32;
  sub_C7D4E0((unsigned __int8 *)v25, &src);
  v15 = src;
  v16 = n;
  v26 = v28;
  if ( (char *)src + n && !src )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v24 = n;
  if ( n > 0xF )
  {
    v26 = (_QWORD *)sub_22409D0((__int64)&v26, &v24, 0);
    v22 = v26;
    v28[0] = v24;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v28[0]) = *(_BYTE *)src;
      v17 = v28;
      goto LABEL_27;
    }
    if ( !n )
    {
      v17 = v28;
      goto LABEL_27;
    }
    v22 = v28;
  }
  memcpy(v22, v15, v16);
  v16 = v24;
  v17 = v26;
LABEL_27:
  v27 = v16;
  *((_BYTE *)v17 + v16) = 0;
  v1 = a1 + 1;
  v18 = (_QWORD *)a1[1];
  if ( v26 == v28 )
  {
    v23 = v27;
    if ( v27 )
    {
      if ( v27 == 1 )
        *(_BYTE *)v18 = v28[0];
      else
        memcpy(v18, v28, v27);
      v23 = v27;
      v18 = (_QWORD *)a1[1];
    }
    a1[2] = v23;
    *((_BYTE *)v18 + v23) = 0;
    v18 = v26;
    goto LABEL_31;
  }
  v19 = v27;
  v20 = v28[0];
  if ( v18 == a1 + 3 )
  {
    a1[1] = v26;
    a1[2] = v19;
    a1[3] = v20;
    goto LABEL_45;
  }
  v21 = a1[3];
  a1[1] = v26;
  a1[2] = v19;
  a1[3] = v20;
  if ( !v18 )
  {
LABEL_45:
    v26 = v28;
    v18 = v28;
    goto LABEL_31;
  }
  v26 = v18;
  v28[0] = v21;
LABEL_31:
  v27 = 0;
  *(_BYTE *)v18 = 0;
  if ( v26 != v28 )
    j_j___libc_free_0((unsigned __int64)v26);
  if ( src != v32 )
    _libc_free((unsigned __int64)src);
  return v1;
}
