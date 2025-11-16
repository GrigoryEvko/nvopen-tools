// Function: sub_1A92D40
// Address: 0x1a92d40
//
__int64 __fastcall sub_1A92D40(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // r14
  __int64 *v19; // rbx
  __int64 *v20; // r15
  __int64 v21; // rsi
  __int64 *v23; // [rsp+8h] [rbp-38h]

  v1 = (__int64 *)a1[1];
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9A488 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_33;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9A488);
  v5 = (__int64 *)a1[1];
  a1[24] = *(_QWORD *)(v4 + 160);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9E06C )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_30;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9E06C);
  v9 = (__int64 *)a1[1];
  a1[25] = v8 + 160;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F9920C )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_31;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F9920C);
  v13 = (__int64 *)a1[1];
  a1[26] = v12 + 160;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9B6E8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_32;
  }
  a1[27] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
             *(_QWORD *)(v14 + 8),
             &unk_4F9B6E8)
         + 360;
  v16 = a1[26];
  v17 = *(__int64 **)(v16 + 32);
  v23 = *(__int64 **)(v16 + 40);
  while ( v23 != v17 )
  {
    v18 = *v17;
    v19 = *(__int64 **)(*v17 + 16);
    if ( *(__int64 **)(*v17 + 8) != v19 )
    {
      v20 = *(__int64 **)(*v17 + 8);
      do
      {
        v21 = *v20++;
        sub_1A92A70((__int64)a1, v21);
      }
      while ( v19 != v20 );
    }
    ++v17;
    sub_1A922B0((__int64)a1, v18);
  }
  return 0;
}
