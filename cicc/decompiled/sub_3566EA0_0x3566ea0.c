// Function: sub_3566EA0
// Address: 0x3566ea0
//
__int64 __fastcall sub_3566EA0(_QWORD *a1, __int64 *a2)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  unsigned __int8 (*v6)(void); // rdx
  unsigned __int8 (*v7)(void); // rdx
  __int64 (*v8)(); // rax
  __int64 (*v9)(); // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 (*v23)(void); // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 *v27; // r13
  __int64 v28; // rsi
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_BB98D0(a1, *a2) )
    return 0;
  if ( !byte_503EEA8 )
    return 0;
  v29[0] = *(_QWORD *)(*a2 + 120);
  if ( (unsigned __int8)sub_A73ED0(v29, 47) )
  {
    if ( !word_503ED4E )
      return 0;
  }
  v4 = a2[2];
  v5 = *(_QWORD *)v4;
  v6 = *(unsigned __int8 (**)(void))(*(_QWORD *)v4 + 272LL);
  if ( (char *)v6 != (char *)sub_3059460 )
  {
    if ( !v6() )
      return 0;
    v5 = *(_QWORD *)a2[2];
  }
  v7 = *(unsigned __int8 (**)(void))(v5 + 384);
  if ( (char *)v7 != (char *)sub_3059490 )
  {
    if ( !v7() )
      goto LABEL_17;
    v5 = *(_QWORD *)a2[2];
  }
  v8 = *(__int64 (**)())(v5 + 216);
  if ( v8 == sub_2F391C0 || !v8() )
    return 0;
  v9 = *(__int64 (**)())(*(_QWORD *)a2[2] + 216LL);
  if ( v9 == sub_2F391C0 )
    BUG();
  if ( !*(_QWORD *)(v9() + 104) )
    return 0;
LABEL_17:
  v10 = (__int64 *)a1[1];
  a1[25] = a2;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_50208AC )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_42;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_50208AC);
  v14 = (__int64 *)a1[1];
  a1[27] = v13 + 200;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_501FE44 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_44;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_501FE44);
  v18 = (__int64 *)a1[1];
  a1[28] = v17 + 200;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_50209AC )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_43;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_50209AC);
  v22 = a1[25];
  a1[26] = *(_QWORD *)(v21 + 200);
  v23 = *(__int64 (**)(void))(**(_QWORD **)(v22 + 16) + 128LL);
  v24 = 0;
  if ( v23 != sub_2DAC790 )
  {
    v24 = v23();
    v22 = a1[25];
  }
  a1[30] = v24;
  sub_2F5FFA0(a1 + 31, v22);
  v25 = a1[27];
  v26 = *(__int64 **)(v25 + 32);
  v27 = *(__int64 **)(v25 + 40);
  if ( v26 == v27 )
    return 0;
  do
  {
    v28 = *v26++;
    sub_3566D90((__int64)a1, v28);
  }
  while ( v27 != v26 );
  return 0;
}
