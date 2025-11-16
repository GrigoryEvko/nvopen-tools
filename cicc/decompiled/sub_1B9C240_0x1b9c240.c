// Function: sub_1B9C240
// Address: 0x1b9c240
//
__int64 __fastcall sub_1B9C240(unsigned int *a1, __int64 *a2, unsigned int a3)
{
  unsigned __int64 *v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *i; // rdx
  _QWORD *v13; // r14
  __int64 v14; // r9
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // rdi
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  unsigned __int64 *v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  int v25; // edx
  unsigned int *v26; // r8
  __int64 v27; // r9
  char v28; // al
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r8
  unsigned __int64 *v32; // rax
  unsigned __int64 *v33; // rsi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdx
  unsigned __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned int *v39; // r8
  int v40; // r9d
  unsigned __int64 *v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // r9
  bool v44; // zf
  unsigned int v45; // ebx
  __int64 **v46; // rax
  __int64 v47; // rax
  unsigned int *v48; // r8
  int v49; // r9d
  __int64 *v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+20h] [rbp-60h]
  __int64 v55; // [rsp+20h] [rbp-60h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+28h] [rbp-58h]
  __int64 v58; // [rsp+28h] [rbp-58h]
  unsigned int *v59; // [rsp+30h] [rbp-50h]
  __int64 v61; // [rsp+38h] [rbp-48h]
  __int64 v62; // [rsp+38h] [rbp-48h]
  __int64 v63; // [rsp+38h] [rbp-48h]
  unsigned __int64 *v64; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 *v65[7]; // [rsp+48h] [rbp-38h] BYREF

  v4 = (unsigned __int64 *)a2;
  v5 = *(_QWORD *)(*((_QWORD *)a1 + 56) + 48LL);
  v6 = *(_QWORD *)(v5 + 112);
  if ( v6 == *(_QWORD *)(v5 + 104) )
    v7 = *(unsigned int *)(v5 + 124);
  else
    v7 = *(unsigned int *)(v5 + 120);
  v8 = (_QWORD *)(v6 + 8 * v7);
  v9 = sub_15CC2D0(v5 + 96, (__int64)a2);
  v10 = *(_QWORD *)(v5 + 112);
  if ( v10 == *(_QWORD *)(v5 + 104) )
    v11 = *(unsigned int *)(v5 + 124);
  else
    v11 = *(unsigned int *)(v5 + 120);
  for ( i = (_QWORD *)(v10 + 8 * v11); i != v9; ++v9 )
  {
    if ( *v9 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  if ( v8 != v9 )
    v4 = (unsigned __int64 *)sub_15A0680(*a2, 1, 0);
  v13 = a1 + 72;
  v64 = v4;
  v65[0] = v4;
  if ( a1 + 74 != (unsigned int *)sub_1B975A0((__int64)(a1 + 72), (unsigned __int64 *)v65)
    && *(_QWORD *)(sub_1B975A0((__int64)(a1 + 72), (unsigned __int64 *)&v64)[5] + 8LL * a3) )
  {
    return *(_QWORD *)(*sub_1B99AC0((_QWORD *)a1 + 36, (unsigned __int64 *)v65) + 8LL * a3);
  }
  v16 = (unsigned __int64 *)*((_QWORD *)a1 + 44);
  v17 = (unsigned __int64 *)(a1 + 86);
  v59 = a1 + 70;
  if ( !v16 )
    goto LABEL_52;
  v18 = (unsigned __int64 *)(a1 + 86);
  v19 = v16;
  do
  {
    while ( 1 )
    {
      v20 = v19[2];
      v21 = v19[3];
      if ( v19[4] >= (unsigned __int64)v4 )
        break;
      v19 = (unsigned __int64 *)v19[3];
      if ( !v21 )
        goto LABEL_19;
    }
    v18 = v19;
    v19 = (unsigned __int64 *)v19[2];
  }
  while ( v20 );
LABEL_19:
  if ( v17 == v18 || v18[4] > (unsigned __int64)v4 )
  {
LABEL_52:
    v27 = (*(__int64 (__fastcall **)(unsigned int *, unsigned __int64 *))(*(_QWORD *)a1 + 16LL))(a1, v4);
LABEL_53:
    v58 = v27;
    sub_1B99BD0(v59, (unsigned __int64)v4, a3, v27, v26, v27);
    return v58;
  }
  v64 = v4;
  v22 = (unsigned __int64 *)(a1 + 86);
  do
  {
    while ( 1 )
    {
      v23 = v16[2];
      v24 = v16[3];
      if ( v16[4] >= (unsigned __int64)v4 )
        break;
      v16 = (unsigned __int64 *)v16[3];
      if ( !v24 )
        goto LABEL_25;
    }
    v22 = v16;
    v16 = (unsigned __int64 *)v16[2];
  }
  while ( v23 );
LABEL_25:
  if ( v17 == v22 || v22[4] > (unsigned __int64)v4 )
  {
    v65[0] = (unsigned __int64 *)&v64;
    v22 = sub_1B99EB0((_QWORD *)a1 + 42, v22, v65);
  }
  v25 = a1[22];
  v26 = (unsigned int *)(48LL * a3);
  v27 = **(_QWORD **)((char *)v26 + v22[5]);
  if ( v25 == 1 )
    goto LABEL_53;
  v54 = **(_QWORD **)(v22[5] + 48LL * a3);
  v28 = sub_1B960F0(*((_QWORD *)a1 + 57), (__int64)v4, v25);
  v29 = 0;
  v30 = v54;
  v31 = 48LL * a3;
  if ( !v28 )
    v29 = 8LL * (a1[22] - 1);
  v32 = (unsigned __int64 *)*((_QWORD *)a1 + 44);
  v64 = v4;
  v33 = (unsigned __int64 *)(a1 + 86);
  if ( !v32 )
    goto LABEL_38;
  do
  {
    while ( 1 )
    {
      v34 = v32[2];
      v35 = v32[3];
      if ( v32[4] >= (unsigned __int64)v4 )
        break;
      v32 = (unsigned __int64 *)v32[3];
      if ( !v35 )
        goto LABEL_36;
    }
    v33 = v32;
    v32 = (unsigned __int64 *)v32[2];
  }
  while ( v34 );
LABEL_36:
  if ( v17 == v33 || v33[4] > (unsigned __int64)v4 )
  {
LABEL_38:
    v52 = v54;
    v55 = v29;
    v65[0] = (unsigned __int64 *)&v64;
    v36 = sub_1B99EB0((_QWORD *)a1 + 42, v33, v65);
    v31 = 48LL * a3;
    v30 = v52;
    v29 = v55;
    v33 = v36;
  }
  v53 = v30;
  v51 = *((_QWORD *)a1 + 14);
  v37 = *(_QWORD *)(*(_QWORD *)(v33[5] + v31) + v29);
  v56 = *((_QWORD *)a1 + 13);
  if ( !v37 )
    BUG();
  v38 = *(_QWORD *)(v37 + 32);
  v50 = (__int64 *)(a1 + 24);
  if ( v38 )
    v38 -= 24;
  sub_17050D0((__int64 *)a1 + 12, v38);
  if ( sub_1B960F0(*((_QWORD *)a1 + 57), (__int64)v4, a1[22]) )
  {
    v57 = (*(__int64 (__fastcall **)(unsigned int *, __int64))(*(_QWORD *)a1 + 16LL))(a1, v53);
    sub_1B99BD0(v59, (unsigned __int64)v4, a3, v57, v39, v40);
    v14 = v57;
  }
  else
  {
    v45 = 0;
    v46 = (__int64 **)sub_16463B0((__int64 *)*v4, a1[22]);
    v47 = sub_1599EF0(v46);
    sub_1B99BD0(v59, (unsigned __int64)v4, a3, v47, v48, v49);
    if ( a1[22] )
    {
      do
      {
        v65[0] = (unsigned __int64 *)__PAIR64__(v45++, a3);
        sub_1B99F70((__int64 *)a1, v4, (unsigned int *)v65);
      }
      while ( a1[22] > v45 );
      v13 = a1 + 72;
    }
    v65[0] = v4;
    v14 = *(_QWORD *)(*sub_1B99AC0(v13, (unsigned __int64 *)v65) + 8LL * a3);
  }
  if ( v56 )
  {
    *((_QWORD *)a1 + 13) = v56;
    *((_QWORD *)a1 + 14) = v51;
    if ( v51 != v56 + 40 )
    {
      if ( !v51 )
        BUG();
      v41 = *(unsigned __int64 **)(v51 + 24);
      v65[0] = v41;
      if ( v41 )
      {
        v61 = v14;
        sub_1623A60((__int64)v65, (__int64)v41, 2);
        v14 = v61;
      }
      v62 = v14;
      sub_17CD270(v50);
      v42 = (unsigned __int8 *)v65[0];
      v43 = v62;
      v44 = v65[0] == 0;
      *((unsigned __int64 **)a1 + 12) = v65[0];
      if ( !v44 )
      {
        sub_1623210((__int64)v65, v42, (__int64)v50);
        v65[0] = 0;
        v43 = v62;
      }
      v63 = v43;
      sub_17CD270((__int64 *)v65);
      return v63;
    }
  }
  else
  {
    *((_QWORD *)a1 + 13) = 0;
    *((_QWORD *)a1 + 14) = 0;
  }
  return v14;
}
