// Function: sub_26BEF90
// Address: 0x26bef90
//
_QWORD *__fastcall sub_26BEF90(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r12
  __int64 i; // r12
  _QWORD *v6; // r13
  _QWORD *v7; // r15
  unsigned __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // r8
  __int64 v11; // rcx
  int *v12; // rsi
  size_t v13; // r13
  int *v14; // rsi
  size_t v15; // r15
  __int64 v16; // r15
  _QWORD *v17; // r14
  __int64 v18; // rbx
  _QWORD *v19; // r12
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // r8
  __int64 v27; // rax
  bool v28; // cc
  unsigned __int64 v29; // r14
  int *v30; // rsi
  size_t v31; // r12
  size_t v32; // rdx
  __int64 v35; // [rsp+8h] [rbp-128h]
  __int64 v37; // [rsp+20h] [rbp-110h]
  __int64 v38; // [rsp+20h] [rbp-110h]
  __int64 v40; // [rsp+38h] [rbp-F8h]
  __int64 v41; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v42; // [rsp+38h] [rbp-F8h]
  size_t v43; // [rsp+38h] [rbp-F8h]
  _QWORD *v44; // [rsp+40h] [rbp-F0h]
  __int64 v45; // [rsp+40h] [rbp-F0h]
  int *v46; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v47; // [rsp+48h] [rbp-E8h]
  __int64 v48; // [rsp+48h] [rbp-E8h]
  _QWORD v49[2]; // [rsp+50h] [rbp-E0h] BYREF
  int v50[52]; // [rsp+60h] [rbp-D0h] BYREF

  v4 = a1;
  v35 = a3 & 1;
  v37 = (a3 - 1) / 2;
  if ( a2 >= v37 )
  {
    v17 = (_QWORD *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_46;
    v16 = a2;
    goto LABEL_32;
  }
  for ( i = a2; ; i = v11 )
  {
    v6 = *(_QWORD **)(a1 + 16 * (i + 1));
    v40 = 2 * (i + 1) - 1;
    v7 = *(_QWORD **)(a1 + 8 * v40);
    v47 = sub_EF9210(v6);
    v8 = sub_EF9210(v7);
    v9 = v47 <= v8;
    v10 = (_QWORD *)(a1 + 8 * v40);
    v11 = v40;
    if ( v47 == v8 )
    {
      v12 = (int *)v6[2];
      v13 = v6[3];
      if ( v12 )
      {
        sub_C7D030(v50);
        sub_C7D280(v50, v12, v13);
        sub_C7D290(v50, v49);
        v13 = v49[0];
        v11 = 2 * (i + 1) - 1;
        v10 = (_QWORD *)(a1 + 8 * v40);
      }
      v14 = (int *)v7[2];
      v15 = v7[3];
      if ( v14 )
      {
        v41 = v11;
        v44 = v10;
        sub_C7D030(v50);
        sub_C7D280(v50, v14, v15);
        sub_C7D290(v50, v49);
        v15 = v49[0];
        v11 = v41;
        v10 = v44;
      }
      v9 = v15 <= v13;
    }
    if ( v9 )
    {
      v11 = 2 * (i + 1);
      v10 = (_QWORD *)(a1 + 16 * (i + 1));
    }
    *(_QWORD *)(a1 + 8 * i) = *v10;
    if ( v11 >= v37 )
      break;
  }
  v4 = a1;
  v16 = v11;
  v17 = v10;
  if ( !v35 )
  {
LABEL_32:
    if ( (a3 - 2) / 2 == v16 )
    {
      v16 = 2 * v16 + 1;
      *v17 = *(_QWORD *)(v4 + 8 * v16);
      v17 = (_QWORD *)(v4 + 8 * v16);
    }
  }
  v18 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    v48 = v4;
    while ( 1 )
    {
      v19 = *(_QWORD **)(v48 + 8 * v18);
      v20 = sub_EF9210(v19);
      if ( unk_4F838D3 )
      {
        v21 = a4[8];
        if ( v21 )
        {
LABEL_26:
          v28 = v20 <= v21;
          if ( v20 != v21 )
            goto LABEL_27;
          goto LABEL_36;
        }
      }
      v22 = a4[20];
      if ( !a4[14] )
        break;
      v23 = a4[12];
      if ( v22 )
      {
        v24 = a4[18];
        v25 = *(_DWORD *)(v24 + 32);
        if ( *(_DWORD *)(v23 + 32) >= v25
          && (*(_DWORD *)(v23 + 32) != v25 || *(_DWORD *)(v23 + 36) >= *(_DWORD *)(v24 + 36)) )
        {
          goto LABEL_22;
        }
      }
      v21 = *(_QWORD *)(v23 + 40);
LABEL_25:
      if ( v21 )
        goto LABEL_26;
LABEL_35:
      v29 = a4[7] != 0;
      v28 = v20 <= v29;
      if ( v20 != v29 )
      {
LABEL_27:
        if ( v28 )
          goto LABEL_45;
        goto LABEL_28;
      }
LABEL_36:
      v30 = (int *)v19[2];
      v31 = v19[3];
      if ( v30 )
      {
        sub_C7D030(v50);
        sub_C7D280(v50, v30, v31);
        sub_C7D290(v50, v49);
        v31 = v49[0];
      }
      v32 = a4[3];
      v46 = (int *)a4[2];
      if ( v46 )
      {
        v43 = a4[3];
        sub_C7D030(v50);
        sub_C7D280(v50, v46, v43);
        sub_C7D290(v50, v49);
        v32 = v49[0];
      }
      if ( v32 <= v31 )
      {
LABEL_45:
        v17 = (_QWORD *)(v48 + 8 * v16);
        goto LABEL_46;
      }
      v19 = *(_QWORD **)(v48 + 8 * v18);
LABEL_28:
      *(_QWORD *)(v48 + 8 * v16) = v19;
      v16 = v18;
      if ( a2 >= v18 )
      {
        v17 = (_QWORD *)(v48 + 8 * v18);
        goto LABEL_46;
      }
      v18 = (v18 - 1) / 2;
    }
    if ( !v22 )
      goto LABEL_35;
    v24 = a4[18];
LABEL_22:
    v26 = *(_QWORD *)(v24 + 64);
    v38 = v24 + 48;
    if ( v26 == v24 + 48 )
      goto LABEL_35;
    v21 = 0;
    do
    {
      v42 = v20;
      v45 = v26;
      v21 += sub_EF9210((_QWORD *)(v26 + 48));
      v27 = sub_220EF30(v45);
      v20 = v42;
      v26 = v27;
    }
    while ( v38 != v27 );
    goto LABEL_25;
  }
LABEL_46:
  *v17 = a4;
  return a4;
}
