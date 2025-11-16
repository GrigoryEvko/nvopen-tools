// Function: sub_1905440
// Address: 0x1905440
//
__int64 __fastcall sub_1905440(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r14
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // rax
  int v16; // edi
  int v17; // edi
  __int64 v18; // r15
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // rsi
  unsigned int v22; // esi
  unsigned int v23; // esi
  unsigned int v24; // eax
  __int64 v25; // rax
  unsigned int v26; // eax
  unsigned int v27; // eax
  const void **v28; // rsi
  __int64 v29; // rdi
  __int64 i; // r14
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]

  v6 = a1[1];
  v7 = *a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - *a1) >> 3);
  if ( v8 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  v10 = a2;
  if ( v8 )
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - v7) >> 3);
  v11 = __CFADD__(v9, v8);
  v12 = v9 - 0x3333333333333333LL * ((v6 - v7) >> 3);
  v13 = a2 - v7;
  if ( v11 )
  {
    v34 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v36 = 0;
      v14 = 40;
      v43 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v34 = 40 * v12;
  }
  v39 = a2;
  v42 = a2 - v7;
  v35 = sub_22077B0(v34);
  v13 = v42;
  a2 = v39;
  v43 = v35;
  v36 = v35 + v34;
  v14 = v35 + 40;
LABEL_7:
  v15 = v43 + v13;
  if ( v43 + v13 )
  {
    *(_QWORD *)v15 = *(_QWORD *)a3;
    v16 = *(_DWORD *)(a3 + 16);
    *(_DWORD *)(a3 + 16) = 0;
    *(_DWORD *)(v15 + 16) = v16;
    *(_QWORD *)(v15 + 8) = *(_QWORD *)(a3 + 8);
    v17 = *(_DWORD *)(a3 + 32);
    *(_DWORD *)(a3 + 32) = 0;
    *(_DWORD *)(v15 + 32) = v17;
    *(_QWORD *)(v15 + 24) = *(_QWORD *)(a3 + 24);
  }
  if ( a2 != v7 )
  {
    v18 = v43;
    v19 = v7;
    while ( 1 )
    {
      if ( !v18 )
        goto LABEL_13;
      *(_QWORD *)v18 = *(_QWORD *)v19;
      v22 = *(_DWORD *)(v19 + 16);
      *(_DWORD *)(v18 + 16) = v22;
      if ( v22 <= 0x40 )
      {
        *(_QWORD *)(v18 + 8) = *(_QWORD *)(v19 + 8);
        v20 = *(_DWORD *)(v19 + 32);
        *(_DWORD *)(v18 + 32) = v20;
        if ( v20 <= 0x40 )
          goto LABEL_12;
      }
      else
      {
        v37 = a2;
        v40 = v19;
        sub_16A4FD0(v18 + 8, (const void **)(v19 + 8));
        v19 = v40;
        a2 = v37;
        v23 = *(_DWORD *)(v40 + 32);
        *(_DWORD *)(v18 + 32) = v23;
        if ( v23 <= 0x40 )
        {
LABEL_12:
          *(_QWORD *)(v18 + 24) = *(_QWORD *)(v19 + 24);
LABEL_13:
          v19 += 40;
          v21 = v18 + 40;
          if ( a2 == v19 )
            goto LABEL_19;
          goto LABEL_14;
        }
      }
      v38 = a2;
      v41 = v19;
      sub_16A4FD0(v18 + 24, (const void **)(v19 + 24));
      a2 = v38;
      v21 = v18 + 40;
      v19 = v41 + 40;
      if ( v38 == v41 + 40 )
      {
LABEL_19:
        v14 = v18 + 80;
        break;
      }
LABEL_14:
      v18 = v21;
    }
  }
  if ( a2 != v6 )
  {
    while ( 1 )
    {
      *(_QWORD *)v14 = *(_QWORD *)v10;
      v26 = *(_DWORD *)(v10 + 16);
      *(_DWORD *)(v14 + 16) = v26;
      if ( v26 <= 0x40 )
      {
        *(_QWORD *)(v14 + 8) = *(_QWORD *)(v10 + 8);
        v24 = *(_DWORD *)(v10 + 32);
        *(_DWORD *)(v14 + 32) = v24;
        if ( v24 <= 0x40 )
          goto LABEL_23;
LABEL_26:
        v28 = (const void **)(v10 + 24);
        v29 = v14 + 24;
        v10 += 40;
        v14 += 40;
        sub_16A4FD0(v29, v28);
        if ( v6 == v10 )
          break;
      }
      else
      {
        sub_16A4FD0(v14 + 8, (const void **)(v10 + 8));
        v27 = *(_DWORD *)(v10 + 32);
        *(_DWORD *)(v14 + 32) = v27;
        if ( v27 > 0x40 )
          goto LABEL_26;
LABEL_23:
        v25 = *(_QWORD *)(v10 + 24);
        v10 += 40;
        v14 += 40;
        *(_QWORD *)(v14 - 16) = v25;
        if ( v6 == v10 )
          break;
      }
    }
  }
  for ( i = v7; v6 != i; i += 40 )
  {
    if ( *(_DWORD *)(i + 32) > 0x40u )
    {
      v31 = *(_QWORD *)(i + 24);
      if ( v31 )
        j_j___libc_free_0_0(v31);
    }
    if ( *(_DWORD *)(i + 16) > 0x40u )
    {
      v32 = *(_QWORD *)(i + 8);
      if ( v32 )
        j_j___libc_free_0_0(v32);
    }
  }
  if ( v7 )
    j_j___libc_free_0(v7, a1[2] - v7);
  a1[1] = v14;
  *a1 = v43;
  a1[2] = v36;
  return v36;
}
