// Function: sub_C1BBC0
// Address: 0xc1bbc0
//
_QWORD *__fastcall sub_C1BBC0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, __int64 a5, _QWORD *a6)
{
  unsigned int *v6; // r15
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // r14
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // r11
  _QWORD *v22; // r9
  unsigned __int64 v24; // r8
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // r11
  __int64 v33; // rdi
  unsigned __int64 v34; // rbx
  _QWORD *v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // [rsp+8h] [rbp-118h]
  __int64 v39; // [rsp+10h] [rbp-110h]
  _QWORD *v42; // [rsp+20h] [rbp-100h]
  __int64 v43; // [rsp+28h] [rbp-F8h]
  _QWORD *v44; // [rsp+30h] [rbp-F0h]
  _QWORD *v45; // [rsp+30h] [rbp-F0h]
  __int64 v46; // [rsp+38h] [rbp-E8h]
  __int64 v47; // [rsp+38h] [rbp-E8h]
  _QWORD *v48; // [rsp+38h] [rbp-E8h]
  __int64 v49; // [rsp+38h] [rbp-E8h]
  _QWORD v50[2]; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+58h] [rbp-C8h]
  char v53; // [rsp+60h] [rbp-C0h]

  v6 = (unsigned int *)a2;
  v8 = sub_C16140(__PAIR128__(a4, a3), (__int64)"selected", 8);
  v9 = *(_QWORD **)(a1 + 168);
  v39 = v8;
  v10 = v8;
  v12 = v11;
  v43 = v11;
  if ( v9 )
  {
    v13 = sub_C1BA30(v9, a2);
    if ( v13 )
      v6 = (unsigned int *)(v13 + 16);
  }
  v14 = a1 + 128;
  v15 = *(_QWORD *)(a1 + 136);
  if ( !v15 )
    return 0;
  v16 = *v6;
  v17 = v14;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v15 + 32) < v16 )
      {
        v15 = *(_QWORD *)(v15 + 24);
        goto LABEL_10;
      }
      if ( *(_DWORD *)(v15 + 32) == v16 && *(_DWORD *)(v15 + 36) < v6[1] )
        break;
      v17 = v15;
      v15 = *(_QWORD *)(v15 + 16);
      if ( !v15 )
        goto LABEL_11;
    }
    v15 = *(_QWORD *)(v15 + 24);
LABEL_10:
    ;
  }
  while ( v15 );
LABEL_11:
  if ( v14 == v17 || *(_DWORD *)(v17 + 32) > v16 || *(_DWORD *)(v17 + 32) == v16 && v6[1] < *(_DWORD *)(v17 + 36) )
    return 0;
  if ( v12 )
  {
    v18 = v12;
    v19 = v10;
    if ( unk_4F838D1 )
    {
      v18 = sub_B2F650(v10, v12);
      v19 = 0;
    }
  }
  else
  {
    v18 = 0;
    v19 = v10;
  }
  v51 = v19;
  v52 = v18;
  v20 = sub_C1BAB0(v17 + 40, (__int64)&v51);
  v21 = v17 + 40;
  v22 = v20;
  if ( v20 != (_QWORD *)(v17 + 48) )
    return v20 + 6;
  if ( a6 && a6[3] )
  {
    if ( v10 )
    {
      v38 = v20;
      sub_C7D030(&v51);
      sub_C7D280(&v51, v10, v12);
      sub_C7D290(&v51, v50);
      v12 = v50[0];
      v22 = v38;
      v21 = v17 + 40;
    }
    v24 = a6[1];
    v25 = *(_QWORD **)(*a6 + 8 * (v12 % v24));
    if ( v25 )
    {
      v26 = (_QWORD *)*v25;
      if ( *(_QWORD *)(*v25 + 8LL) != v12 )
      {
        do
        {
          v27 = (_QWORD *)*v26;
          if ( !*v26 )
            goto LABEL_41;
          v25 = v26;
          if ( v12 % v24 != v27[1] % v24 )
            goto LABEL_41;
          v26 = (_QWORD *)*v26;
        }
        while ( v27[1] != v12 );
      }
      v28 = *v25;
      if ( !*v25 )
        goto LABEL_41;
      v39 = *(_QWORD *)(v28 + 16);
      if ( !v39 )
      {
        v43 = 0;
        goto LABEL_38;
      }
      v43 = *(_QWORD *)(v28 + 24);
      if ( v43 && unk_4F838D1 )
      {
        v45 = v22;
        v49 = v21;
        v29 = sub_B2F650(v39, v43);
        v21 = v49;
        v22 = v45;
        v30 = 0;
      }
      else
      {
LABEL_38:
        v29 = v43;
        v30 = v39;
      }
      v42 = v22;
      v46 = v21;
      v51 = v30;
      v52 = v29;
      v31 = sub_C1BAB0(v21, (__int64)&v51);
      v22 = v42;
      v21 = v46;
      if ( v42 == v31 )
        goto LABEL_41;
      return v31 + 6;
    }
  }
LABEL_41:
  v47 = v21;
  if ( a5 )
  {
    v44 = v22;
    sub_C21B60(&v51, a5, v39, v43);
    v22 = v44;
    v32 = v47;
    if ( v53 )
    {
      v36 = v51;
      v37 = v52;
      if ( v52 && unk_4F838D1 )
      {
        v37 = sub_B2F650(v51, v52);
        v22 = v44;
        v32 = v47;
        v36 = 0;
      }
      v48 = v22;
      v50[0] = v36;
      v50[1] = v37;
      v31 = sub_C1BAB0(v32, (__int64)v50);
      v22 = v48;
      if ( v48 != v31 )
        return v31 + 6;
    }
  }
  if ( v43 )
    return 0;
  v33 = *(_QWORD *)(v17 + 64);
  if ( v22 == (_QWORD *)v33 )
    return 0;
  v34 = 0;
  v35 = v22;
  do
  {
    if ( *(_QWORD *)(v33 + 104) >= v34 )
    {
      v15 = v33 + 48;
      v34 = *(_QWORD *)(v33 + 104);
    }
    v33 = sub_220EF30(v33);
  }
  while ( v35 != (_QWORD *)v33 );
  return (_QWORD *)v15;
}
