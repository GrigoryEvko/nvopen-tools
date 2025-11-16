// Function: sub_D8A980
// Address: 0xd8a980
//
__int64 *__fastcall sub_D8A980(__int64 *a1, __int64 a2, __int64 *a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r14
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // r11
  unsigned int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // rsi
  unsigned int v24; // esi
  unsigned int v25; // esi
  unsigned int v26; // eax
  __int64 v27; // rax
  unsigned int v28; // eax
  unsigned int v29; // eax
  const void **v30; // rsi
  __int64 v31; // rdi
  __int64 i; // r13
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v36; // r15
  __int64 v37; // rax
  unsigned int v38; // eax
  __int64 v39; // [rsp+8h] [rbp-68h]
  _QWORD *v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+18h] [rbp-58h]
  __int64 v42; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+30h] [rbp-40h]
  __int64 v45; // [rsp+30h] [rbp-40h]
  __int64 v46; // [rsp+30h] [rbp-40h]
  __int64 v47; // [rsp+38h] [rbp-38h]

  v7 = a1[1];
  v8 = *a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((v7 - *a1) >> 4);
  if ( v9 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  v11 = a2;
  if ( v9 )
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v7 - v8) >> 4);
  v12 = __CFADD__(v10, v9);
  v13 = v10 - 0x5555555555555555LL * ((v7 - v8) >> 4);
  v14 = a2 - v8;
  if ( v12 )
  {
    v36 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v13 )
    {
      v42 = 0;
      v15 = 48;
      v47 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x2AAAAAAAAAAAAAALL )
      v13 = 0x2AAAAAAAAAAAAAALL;
    v36 = 48 * v13;
  }
  v39 = a5;
  v40 = a4;
  v37 = sub_22077B0(v36);
  v14 = a2 - v8;
  a4 = v40;
  a5 = v39;
  v47 = v37;
  v42 = v37 + v36;
  v15 = v37 + 48;
LABEL_7:
  v16 = *a3;
  v17 = v47 + v14;
  if ( v17 )
  {
    *(_QWORD *)(v17 + 8) = *a4;
    v18 = *(_DWORD *)(a5 + 8);
    *(_QWORD *)v17 = v16;
    *(_DWORD *)(v17 + 24) = v18;
    if ( v18 > 0x40 )
    {
      v41 = v17;
      v46 = a5;
      sub_C43780(v17 + 16, (const void **)a5);
      a5 = v46;
      v17 = v41;
      v38 = *(_DWORD *)(v46 + 24);
      *(_DWORD *)(v41 + 40) = v38;
      if ( v38 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)(v17 + 16) = *(_QWORD *)a5;
      v19 = *(_DWORD *)(a5 + 24);
      *(_DWORD *)(v17 + 40) = v19;
      if ( v19 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(v17 + 32) = *(_QWORD *)(a5 + 16);
        goto LABEL_11;
      }
    }
    sub_C43780(v17 + 32, (const void **)(a5 + 16));
  }
LABEL_11:
  if ( a2 == v8 )
    goto LABEL_22;
  v20 = v47;
  v21 = v8;
  while ( !v20 )
  {
LABEL_15:
    v21 += 48;
    v23 = v20 + 48;
    if ( a2 == v21 )
      goto LABEL_21;
LABEL_16:
    v20 = v23;
  }
  *(_QWORD *)v20 = *(_QWORD *)v21;
  *(_QWORD *)(v20 + 8) = *(_QWORD *)(v21 + 8);
  v24 = *(_DWORD *)(v21 + 24);
  *(_DWORD *)(v20 + 24) = v24;
  if ( v24 <= 0x40 )
  {
    *(_QWORD *)(v20 + 16) = *(_QWORD *)(v21 + 16);
    v22 = *(_DWORD *)(v21 + 40);
    *(_DWORD *)(v20 + 40) = v22;
    if ( v22 > 0x40 )
      goto LABEL_20;
    goto LABEL_14;
  }
  v44 = v21;
  sub_C43780(v20 + 16, (const void **)(v21 + 16));
  v21 = v44;
  v25 = *(_DWORD *)(v44 + 40);
  *(_DWORD *)(v20 + 40) = v25;
  if ( v25 <= 0x40 )
  {
LABEL_14:
    *(_QWORD *)(v20 + 32) = *(_QWORD *)(v21 + 32);
    goto LABEL_15;
  }
LABEL_20:
  v45 = v21;
  sub_C43780(v20 + 32, (const void **)(v21 + 32));
  v23 = v20 + 48;
  v21 = v45 + 48;
  if ( a2 != v45 + 48 )
    goto LABEL_16;
LABEL_21:
  v15 = v20 + 96;
LABEL_22:
  if ( a2 != v7 )
  {
    while ( 1 )
    {
      *(_QWORD *)v15 = *(_QWORD *)v11;
      *(_QWORD *)(v15 + 8) = *(_QWORD *)(v11 + 8);
      v28 = *(_DWORD *)(v11 + 24);
      *(_DWORD *)(v15 + 24) = v28;
      if ( v28 <= 0x40 )
      {
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v11 + 16);
        v26 = *(_DWORD *)(v11 + 40);
        *(_DWORD *)(v15 + 40) = v26;
        if ( v26 <= 0x40 )
          goto LABEL_25;
LABEL_28:
        v30 = (const void **)(v11 + 32);
        v31 = v15 + 32;
        v11 += 48;
        v15 += 48;
        sub_C43780(v31, v30);
        if ( v7 == v11 )
          break;
      }
      else
      {
        sub_C43780(v15 + 16, (const void **)(v11 + 16));
        v29 = *(_DWORD *)(v11 + 40);
        *(_DWORD *)(v15 + 40) = v29;
        if ( v29 > 0x40 )
          goto LABEL_28;
LABEL_25:
        v27 = *(_QWORD *)(v11 + 32);
        v11 += 48;
        v15 += 48;
        *(_QWORD *)(v15 - 16) = v27;
        if ( v7 == v11 )
          break;
      }
    }
  }
  for ( i = v8; v7 != i; i += 48 )
  {
    if ( *(_DWORD *)(i + 40) > 0x40u )
    {
      v33 = *(_QWORD *)(i + 32);
      if ( v33 )
        j_j___libc_free_0_0(v33);
    }
    if ( *(_DWORD *)(i + 24) > 0x40u )
    {
      v34 = *(_QWORD *)(i + 16);
      if ( v34 )
        j_j___libc_free_0_0(v34);
    }
  }
  if ( v8 )
    j_j___libc_free_0(v8, a1[2] - v8);
  *a1 = v47;
  a1[1] = v15;
  a1[2] = v42;
  return a1;
}
