// Function: sub_3376670
// Address: 0x3376670
//
unsigned __int64 *__fastcall sub_3376670(
        unsigned __int64 *a1,
        __int64 a2,
        _QWORD *a3,
        _QWORD *a4,
        __int64 *a5,
        _DWORD *a6)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r15
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // r11
  __int64 v18; // r9
  unsigned __int8 *v19; // rsi
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // rbx
  __int64 v22; // r13
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r12
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rax
  unsigned __int64 i; // r14
  __int64 v29; // rsi
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 *v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+8h] [rbp-78h]
  _QWORD *v36; // [rsp+8h] [rbp-78h]
  _QWORD *v37; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  unsigned __int64 v40; // [rsp+20h] [rbp-60h]
  unsigned __int64 v41; // [rsp+28h] [rbp-58h]
  unsigned __int64 v43; // [rsp+38h] [rbp-48h]
  __int64 v44[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = a1[1];
  v7 = *a1;
  v8 = (__int64)(v6 - *a1) >> 5;
  if ( v8 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  v10 = a2;
  if ( v8 )
    v9 = (__int64)(v6 - v7) >> 5;
  v11 = a2;
  v12 = __CFADD__(v9, v8);
  v13 = v9 + v8;
  v14 = a2 - v7;
  if ( v12 )
  {
    v31 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v13 )
    {
      v41 = 0;
      v15 = 32;
      v43 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x3FFFFFFFFFFFFFFLL )
      v13 = 0x3FFFFFFFFFFFFFFLL;
    v31 = 32 * v13;
  }
  v34 = a5;
  v36 = a4;
  v37 = a3;
  v32 = sub_22077B0(v31);
  v14 = a2 - v7;
  a3 = v37;
  a4 = v36;
  a5 = v34;
  v43 = v32;
  v41 = v32 + v31;
  v15 = v32 + 32;
LABEL_7:
  v16 = *a5;
  v17 = *a3;
  v18 = v43 + v14;
  v38 = *a4;
  v44[0] = v16;
  if ( v16 )
  {
    v33 = v17;
    v35 = v18;
    sub_B96E90((__int64)v44, v16, 1);
    v19 = (unsigned __int8 *)v44[0];
    if ( v35 )
    {
      *(_QWORD *)(v35 + 8) = v33;
      *(_QWORD *)(v35 + 24) = v19;
      *(_DWORD *)v35 = *a6;
      *(_QWORD *)(v35 + 16) = v38;
      if ( v19 )
        sub_B976B0((__int64)v44, v19, v35 + 24);
    }
    else if ( v44[0] )
    {
      sub_B91220((__int64)v44, v44[0]);
    }
  }
  else if ( v18 )
  {
    *(_DWORD *)v18 = *a6;
    *(_QWORD *)(v18 + 8) = v17;
    *(_QWORD *)(v18 + 16) = v38;
    *(_QWORD *)(v18 + 24) = 0;
  }
  if ( v10 != v7 )
  {
    v20 = v7;
    v40 = v7;
    v21 = v43;
    v22 = v10;
    v23 = v6;
    v24 = v20;
    while ( 1 )
    {
      if ( v21 )
      {
        *(_DWORD *)v21 = *(_DWORD *)v24;
        *(_QWORD *)(v21 + 8) = *(_QWORD *)(v24 + 8);
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v24 + 16);
        v25 = *(_QWORD *)(v24 + 24);
        *(_QWORD *)(v21 + 24) = v25;
        if ( v25 )
          sub_B96E90(v21 + 24, v25, 1);
      }
      v24 += 32LL;
      if ( v22 == v24 )
        break;
      v21 += 32LL;
    }
    v6 = v23;
    v10 = v22;
    v7 = v40;
    v15 = v21 + 64;
  }
  if ( v10 != v6 )
  {
    do
    {
      v26 = *(_QWORD *)(v11 + 24);
      *(_DWORD *)v15 = *(_DWORD *)v11;
      v27 = *(_QWORD *)(v11 + 8);
      *(_QWORD *)(v15 + 24) = v26;
      *(_QWORD *)(v15 + 8) = v27;
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v11 + 16);
      if ( v26 )
        sub_B96E90(v15 + 24, v26, 1);
      v11 += 32;
      v15 += 32;
    }
    while ( v6 != v11 );
  }
  for ( i = v7; i != v6; i += 32LL )
  {
    v29 = *(_QWORD *)(i + 24);
    if ( v29 )
      sub_B91220(i + 24, v29);
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  *a1 = v43;
  a1[1] = v15;
  a1[2] = v41;
  return a1;
}
