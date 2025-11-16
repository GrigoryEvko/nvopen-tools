// Function: sub_2A65970
// Address: 0x2a65970
//
unsigned __int64 __fastcall sub_2A65970(unsigned __int64 *a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned __int8 *v4; // r13
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int8 *v10; // r15
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int8 *v13; // r9
  __int64 v14; // rbx
  unsigned __int64 v15; // rbx
  unsigned __int8 *v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int8 v18; // si
  unsigned int v19; // esi
  unsigned int v20; // esi
  unsigned __int8 v21; // si
  unsigned __int8 v22; // al
  unsigned int v23; // eax
  unsigned int v24; // eax
  unsigned __int8 v25; // al
  unsigned __int8 *i; // r15
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned int v30; // esi
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int8 *v34; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v35; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v36; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v37; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-48h]
  unsigned __int8 *v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  unsigned __int64 v42; // [rsp+28h] [rbp-38h]

  v4 = (unsigned __int8 *)a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v4[-*a1] >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  v9 = a3;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v4[-v5] >> 3);
  v10 = a2;
  v11 = __CFADD__(v7, v6);
  v12 = v7 - 0x3333333333333333LL * ((__int64)&v4[-v5] >> 3);
  v13 = &a2[-v5];
  if ( v11 )
  {
    v31 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v41 = 0;
      v14 = 40;
      v42 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v31 = 40 * v12;
  }
  v36 = a2;
  v40 = &a2[-v5];
  v32 = sub_22077B0(v31);
  v13 = v40;
  a2 = v36;
  v42 = v32;
  v9 = a3;
  v41 = v32 + v31;
  v14 = v32 + 40;
LABEL_7:
  if ( &v13[v42] )
  {
    v37 = a2;
    sub_22C05A0((__int64)&v13[v42], v9);
    a2 = v37;
  }
  if ( a2 != (unsigned __int8 *)v5 )
  {
    v15 = v42;
    v16 = (unsigned __int8 *)v5;
    while ( 1 )
    {
      if ( !v15 )
        goto LABEL_13;
      v18 = *v16;
      *(_WORD *)v15 = *v16;
      if ( v18 <= 3u )
      {
        if ( v18 > 1u )
          *(_QWORD *)(v15 + 8) = *((_QWORD *)v16 + 1);
LABEL_13:
        v16 += 40;
        v17 = v15 + 40;
        if ( a2 == v16 )
          goto LABEL_22;
        goto LABEL_14;
      }
      if ( (unsigned __int8)(v18 - 4) > 1u )
        goto LABEL_13;
      v19 = *((_DWORD *)v16 + 4);
      *(_DWORD *)(v15 + 16) = v19;
      if ( v19 > 0x40 )
      {
        v34 = a2;
        v38 = v16;
        sub_C43780(v15 + 8, (const void **)v16 + 1);
        v16 = v38;
        a2 = v34;
        v30 = *((_DWORD *)v38 + 8);
        *(_DWORD *)(v15 + 32) = v30;
        if ( v30 <= 0x40 )
        {
LABEL_20:
          *(_QWORD *)(v15 + 24) = *((_QWORD *)v16 + 3);
          goto LABEL_21;
        }
      }
      else
      {
        *(_QWORD *)(v15 + 8) = *((_QWORD *)v16 + 1);
        v20 = *((_DWORD *)v16 + 8);
        *(_DWORD *)(v15 + 32) = v20;
        if ( v20 <= 0x40 )
          goto LABEL_20;
      }
      v35 = a2;
      v39 = v16;
      sub_C43780(v15 + 24, (const void **)v16 + 3);
      a2 = v35;
      v16 = v39;
LABEL_21:
      v21 = v16[1];
      v16 += 40;
      *(_BYTE *)(v15 + 1) = v21;
      v17 = v15 + 40;
      if ( a2 == v16 )
      {
LABEL_22:
        v14 = v15 + 80;
        break;
      }
LABEL_14:
      v15 = v17;
    }
  }
  if ( a2 != v4 )
  {
    while ( 1 )
    {
      v22 = *v10;
      *(_WORD *)v14 = *v10;
      if ( v22 <= 3u )
        break;
      if ( (unsigned __int8)(v22 - 4) > 1u )
      {
LABEL_27:
        v10 += 40;
        v14 += 40;
        if ( v4 == v10 )
          goto LABEL_35;
      }
      else
      {
        v23 = *((_DWORD *)v10 + 4);
        *(_DWORD *)(v14 + 16) = v23;
        if ( v23 > 0x40 )
          sub_C43780(v14 + 8, (const void **)v10 + 1);
        else
          *(_QWORD *)(v14 + 8) = *((_QWORD *)v10 + 1);
        v24 = *((_DWORD *)v10 + 8);
        *(_DWORD *)(v14 + 32) = v24;
        if ( v24 > 0x40 )
          sub_C43780(v14 + 24, (const void **)v10 + 3);
        else
          *(_QWORD *)(v14 + 24) = *((_QWORD *)v10 + 3);
        v25 = v10[1];
        v10 += 40;
        v14 += 40;
        *(_BYTE *)(v14 - 39) = v25;
        if ( v4 == v10 )
          goto LABEL_35;
      }
    }
    if ( v22 > 1u )
      *(_QWORD *)(v14 + 8) = *((_QWORD *)v10 + 1);
    goto LABEL_27;
  }
LABEL_35:
  for ( i = (unsigned __int8 *)v5; i != v4; i += 40 )
  {
    while ( 1 )
    {
      if ( (unsigned int)*i - 4 <= 1 )
      {
        if ( *((_DWORD *)i + 8) > 0x40u )
        {
          v27 = *((_QWORD *)i + 3);
          if ( v27 )
            j_j___libc_free_0_0(v27);
        }
        if ( *((_DWORD *)i + 4) > 0x40u )
        {
          v28 = *((_QWORD *)i + 1);
          if ( v28 )
            break;
        }
      }
      i += 40;
      if ( i == v4 )
        goto LABEL_45;
    }
    j_j___libc_free_0_0(v28);
  }
LABEL_45:
  if ( v5 )
    j_j___libc_free_0(v5);
  a1[1] = v14;
  *a1 = v42;
  a1[2] = v41;
  return v41;
}
