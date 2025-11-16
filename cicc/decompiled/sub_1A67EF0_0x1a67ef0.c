// Function: sub_1A67EF0
// Address: 0x1a67ef0
//
__int64 __fastcall sub_1A67EF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  int *v19; // r14
  __int64 v20; // r12
  unsigned __int8 v21; // bl
  __int64 v22; // rdi
  char v23; // al
  unsigned __int8 v25; // [rsp+7h] [rbp-49h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_35;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  v6 = *(__int64 **)(a1 + 8);
  v26 = v5 + 160;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F9920C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_32;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9920C);
  v10 = *(__int64 **)(a1 + 8);
  v11 = v9 + 160;
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F96DB4 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_33;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F96DB4);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD **)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4FB9E34 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_34;
  }
  v19 = (int *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
                  *(_QWORD *)(v17 + 8),
                  &unk_4FB9E34)
              + 160);
  sub_1C007A0(v19, a2, 1, 1, 1, 0);
  v25 = 0;
  v27 = a2 + 72;
  v20 = *(_QWORD *)(a2 + 80);
  if ( v20 != a2 + 72 )
  {
    do
    {
      v21 = 0;
      do
      {
        v22 = v20 - 24;
        if ( !v20 )
          v22 = 0;
        v23 = sub_1A65DC0(v22, v26, v11, v16, v19);
        v20 = *(_QWORD *)(v20 + 8);
        v21 |= v23;
      }
      while ( v20 != v27 );
      if ( !v21 )
        break;
      v25 = v21;
      v20 = *(_QWORD *)(a2 + 80);
    }
    while ( v20 != v27 );
  }
  return v25;
}
