// Function: sub_23B66B0
// Address: 0x23b66b0
//
__int64 __fastcall sub_23B66B0(__int64 *a1, char a2)
{
  __int64 v3; // r13
  _QWORD *(__fastcall *v4)(__int64 *, __int64); // rax
  _QWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  void *(*v14)(); // rdx
  __int64 v15; // r13
  void *v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rax
  void *(*v20)(); // rdx
  __int64 v21; // r12
  char *v22; // rax
  __int64 v23; // rdx
  void *v24; // rax
  void *(*v25)(); // rax
  void *v26; // rax
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // r13
  char *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // r12
  __int64 *v35; // rax
  __int64 v36; // r12
  _BYTE *v37; // rax
  __int64 v38; // rdx
  __int64 v39[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *a1;
  if ( !*a1 )
    goto LABEL_15;
  v4 = *(_QWORD *(__fastcall **)(__int64 *, __int64))(*(_QWORD *)v3 + 16LL);
  if ( v4 == sub_23AEE80 )
  {
    v5 = (_QWORD *)sub_22077B0(0x68u);
    v9 = v5;
    if ( v5 )
    {
      *v5 = &unk_4A16218;
      sub_C8CD80((__int64)(v5 + 1), (__int64)(v5 + 5), v3 + 8, v6, v7, v8);
      sub_C8CD80((__int64)(v9 + 7), (__int64)(v9 + 11), v3 + 56, v10, v11, v12);
    }
    v39[0] = (__int64)v9;
  }
  else
  {
    v4(v39, v3);
    v9 = (_QWORD *)v39[0];
  }
  if ( !v9 )
    goto LABEL_15;
  v13 = *v9;
  v14 = *(void *(**)())(*v9 + 24LL);
  if ( v14 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D162 )
      goto LABEL_9;
LABEL_14:
    (*(void (__fastcall **)(_QWORD *))(v13 + 8))(v9);
    goto LABEL_15;
  }
  v17 = (void *)((__int64 (__fastcall *)(_QWORD *))v14)(v9);
  v9 = (_QWORD *)v39[0];
  if ( v17 != &unk_4C5D162 )
  {
    if ( !v39[0] )
      goto LABEL_15;
    v13 = *(_QWORD *)v39[0];
    goto LABEL_14;
  }
LABEL_9:
  v15 = v9[1];
  (*(void (__fastcall **)(_QWORD *))(*v9 + 8LL))(v9);
  if ( v15 )
    return v15;
LABEL_15:
  sub_23B2720(v39, a1);
  v18 = (_QWORD *)v39[0];
  if ( !v39[0] )
    goto LABEL_28;
  v19 = *(_QWORD *)v39[0];
  v20 = *(void *(**)())(*(_QWORD *)v39[0] + 24LL);
  if ( v20 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D161 )
      goto LABEL_18;
LABEL_27:
    (*(void (**)(void))(v19 + 8))();
    goto LABEL_28;
  }
  v24 = v20();
  v18 = (_QWORD *)v39[0];
  if ( v24 != &unk_4C5D161 )
  {
    if ( !v39[0] )
      goto LABEL_28;
    v19 = *(_QWORD *)v39[0];
    goto LABEL_27;
  }
LABEL_18:
  v21 = v18[1];
  (*(void (__fastcall **)(_QWORD *))(*v18 + 8LL))(v18);
  if ( v21 )
  {
LABEL_19:
    if ( a2 )
      return *(_QWORD *)(v21 + 40);
    v22 = (char *)sub_BD5D20(v21);
    if ( sub_BC63A0(v22, v23) )
      return *(_QWORD *)(v21 + 40);
    return 0;
  }
LABEL_28:
  sub_23B2720(v39, a1);
  if ( !v39[0]
    || ((v25 = *(void *(**)())(*(_QWORD *)v39[0] + 24LL), v25 != sub_23AE340) ? (v26 = v25()) : (v26 = &unk_4CDFBF8),
        v26 != &unk_4C5D118) )
  {
    sub_23B42E0(v39);
LABEL_42:
    sub_23B2720(v39, a1);
    v33 = (__int64 *)sub_23B5650(v39);
    if ( v33 )
    {
      v34 = *v33;
      sub_23B42E0(v39);
      if ( v34 )
      {
        v21 = *(_QWORD *)(**(_QWORD **)(v34 + 32) + 72LL);
        goto LABEL_19;
      }
    }
    else
    {
      sub_23B42E0(v39);
    }
    sub_23B2720(v39, a1);
    v35 = (__int64 *)sub_23B6650(v39);
    if ( v35 )
    {
      v36 = *v35;
      sub_23B42E0(v39);
      if ( v36 )
      {
        if ( a2 )
          return *(_QWORD *)(*(_QWORD *)v36 + 40LL);
        v37 = (_BYTE *)sub_2E791E0(v36);
        if ( sub_BC63A0(v37, v38) )
          return *(_QWORD *)(*(_QWORD *)v36 + 40LL);
        return 0;
      }
    }
    else
    {
      sub_23B42E0(v39);
    }
    BUG();
  }
  v27 = *(_QWORD *)(v39[0] + 8);
  sub_23B42E0(v39);
  if ( !v27 )
    goto LABEL_42;
  v28 = *(_QWORD *)(v27 + 8);
  v29 = v28 + 8LL * *(unsigned int *)(v27 + 16);
  if ( v28 == v29 )
    return 0;
  v30 = *(_QWORD *)(*(_QWORD *)v28 + 8LL);
  if ( !a2 )
  {
    while ( 1 )
    {
      v28 += 8;
      if ( !sub_B2FC80(v30) )
      {
        v31 = (char *)sub_BD5D20(v30);
        if ( sub_BC63A0(v31, v32) )
          break;
      }
      if ( v28 == v29 )
        return 0;
      v30 = *(_QWORD *)(*(_QWORD *)v28 + 8LL);
    }
  }
  return *(_QWORD *)(v30 + 40);
}
