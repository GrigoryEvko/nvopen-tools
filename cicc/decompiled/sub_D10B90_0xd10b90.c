// Function: sub_D10B90
// Address: 0xd10b90
//
__int64 __fastcall sub_D10B90(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // r15
  __int64 v7; // r10
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r12
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  bool v18; // zf
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 i; // r12
  __int64 v25; // rax
  _QWORD *v26; // rdi
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rsi
  _QWORD *v31; // [rsp+0h] [rbp-60h]
  _QWORD *v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  v6 = a1[1];
  v7 = *a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - *a1) >> 3);
  if ( v8 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  v10 = a2;
  v11 = a2;
  if ( v8 )
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - v7) >> 3);
  v12 = __CFADD__(v9, v8);
  v13 = v9 - 0x3333333333333333LL * ((v6 - v7) >> 3);
  v14 = a2 - v7;
  v15 = v12;
  if ( v12 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v47 = 0;
      v16 = 40;
      goto LABEL_7;
    }
    if ( v13 > 0x333333333333333LL )
      v13 = 0x333333333333333LL;
    v28 = 40 * v13;
  }
  v32 = a4;
  v35 = v10;
  v45 = *a1;
  v29 = sub_22077B0(v28);
  v7 = v45;
  v15 = v29;
  v10 = v35;
  a4 = v32;
  v47 = v28 + v29;
  v16 = v29 + 40;
LABEL_7:
  v17 = v15 + v14;
  if ( v15 + v14 )
  {
    v18 = *(_BYTE *)(a3 + 24) == 0;
    *(_BYTE *)(v17 + 24) = 0;
    if ( !v18 )
    {
      v30 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)v17 = 6;
      *(_QWORD *)(v17 + 8) = 0;
      *(_QWORD *)(v17 + 16) = v30;
      if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
      {
        v31 = a4;
        v33 = v10;
        v36 = v7;
        v40 = v15;
        v46 = v17;
        sub_BD6050((unsigned __int64 *)v17, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
        a4 = v31;
        v10 = v33;
        v7 = v36;
        v15 = v40;
        v17 = v46;
      }
      *(_BYTE *)(v17 + 24) = 1;
    }
    *(_QWORD *)(v17 + 32) = *a4;
  }
  if ( v10 != v7 )
  {
    v19 = v15;
    v20 = v7;
    while ( 1 )
    {
      if ( v19 )
      {
        *(_BYTE *)(v19 + 24) = 0;
        if ( *(_BYTE *)(v20 + 24) )
        {
          *(_QWORD *)v19 = 6;
          *(_QWORD *)(v19 + 8) = 0;
          v21 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(v19 + 16) = v21;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
          {
            v34 = v10;
            v37 = v7;
            v41 = v15;
            sub_BD6050((unsigned __int64 *)v19, *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL);
            v10 = v34;
            v7 = v37;
            v15 = v41;
          }
          *(_BYTE *)(v19 + 24) = 1;
        }
        *(_QWORD *)(v19 + 32) = *(_QWORD *)(v20 + 32);
      }
      v20 += 40;
      if ( v10 == v20 )
        break;
      v19 += 40;
    }
    v16 = v19 + 80;
  }
  if ( v10 != v6 )
  {
    do
    {
      v18 = *(_BYTE *)(v11 + 24) == 0;
      *(_BYTE *)(v16 + 24) = 0;
      if ( !v18 )
      {
        v23 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)v16 = 6;
        *(_QWORD *)(v16 + 8) = 0;
        *(_QWORD *)(v16 + 16) = v23;
        if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        {
          v38 = v7;
          v42 = v15;
          sub_BD6050((unsigned __int64 *)v16, *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v38;
          v15 = v42;
        }
        *(_BYTE *)(v16 + 24) = 1;
      }
      v22 = *(_QWORD *)(v11 + 32);
      v11 += 40;
      v16 += 40;
      *(_QWORD *)(v16 - 8) = v22;
    }
    while ( v6 != v11 );
  }
  for ( i = v7; v6 != i; v15 = v43 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(i + 24) )
      {
        v25 = *(_QWORD *)(i + 16);
        *(_BYTE *)(i + 24) = 0;
        if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
          break;
      }
      i += 40;
      if ( v6 == i )
        goto LABEL_37;
    }
    v26 = (_QWORD *)i;
    i += 40;
    v39 = v7;
    v43 = v15;
    sub_BD60C0(v26);
    v7 = v39;
  }
LABEL_37:
  if ( v7 )
  {
    v44 = v15;
    j_j___libc_free_0(v7, a1[2] - v7);
    v15 = v44;
  }
  a1[1] = v16;
  *a1 = v15;
  a1[2] = v47;
  return v47;
}
