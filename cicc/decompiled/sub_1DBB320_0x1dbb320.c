// Function: sub_1DBB320
// Address: 0x1dbb320
//
__int64 __fastcall sub_1DBB320(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rdx
  __int64 v3; // rax
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  void *v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 i; // rcx
  __int64 v21; // r8
  int v22; // r9d
  bool v23; // zf
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // r8d
  int v35; // r9d

  *(_QWORD *)(a1 + 232) = a2;
  *(_QWORD *)(a1 + 240) = *(_QWORD *)(a2 + 40);
  v2 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v3 = 0;
  if ( v2 != sub_1D00B10 )
  {
    v3 = v2();
    a2 = *(_QWORD *)(a1 + 232);
  }
  *(_QWORD *)(a1 + 248) = v3;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v5 = 0;
  if ( v4 != sub_1D00B00 )
    v5 = v4();
  v6 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 256) = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F96DB4 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_34;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F96DB4);
  v10 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 264) = *(_QWORD *)(v9 + 160);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FCA82C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_36;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FCA82C);
  v14 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 272) = v13;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_35:
    BUG();
  v17 = &unk_4FC62EC;
  while ( *(_UNKNOWN **)v15 != &unk_4FC62EC )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_35;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4FC62EC);
  v23 = *(_QWORD *)(a1 + 288) == 0;
  *(_QWORD *)(a1 + 280) = v18;
  if ( v23 )
  {
    v19 = (__int64 *)sub_22077B0(664);
    if ( v19 )
    {
      memset(v19, 0, 0x298u);
      i = 0;
      v19[12] = (__int64)(v19 + 14);
      v19[17] = (__int64)(v19 + 19);
      v19[18] = 0x1000000000LL;
    }
    *(_QWORD *)(a1 + 288) = v19;
  }
  v24 = *(unsigned int *)(*(_QWORD *)(a1 + 240) + 32LL);
  v25 = *(unsigned int *)(a1 + 408);
  if ( v24 < v25 )
    goto LABEL_24;
  if ( v24 > v25 )
  {
    if ( v24 > *(unsigned int *)(a1 + 412) )
    {
      v17 = (void *)(a1 + 416);
      sub_16CD150(a1 + 400, (const void *)(a1 + 416), *(unsigned int *)(*(_QWORD *)(a1 + 240) + 32LL), 8, v21, v22);
      v25 = *(unsigned int *)(a1 + 408);
    }
    v26 = *(_QWORD *)(a1 + 400);
    v19 = (__int64 *)(v26 + 8 * v24);
    v27 = (__int64 *)(v26 + 8 * v25);
    for ( i = *(_QWORD *)(a1 + 416); v19 != v27; ++v27 )
      *v27 = i;
LABEL_24:
    *(_DWORD *)(a1 + 408) = v24;
  }
  sub_1DBB1A0(a1, (__int64)v17, (__int64)v19, i, v21, v22);
  sub_1DBA310(a1, (__int64)v17, v28, v29, v30, v31);
  sub_1DBAB50(a1, (__int64)v17, v32, v33, v34, v35);
  return 1;
}
