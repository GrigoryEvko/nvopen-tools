// Function: sub_297A4C0
// Address: 0x297a4c0
//
__int64 __fastcall sub_297A4C0(__int64 a1, __int64 a2)
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
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  char v20; // r14
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int8 v24; // bl
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rdi
  char v28; // al
  unsigned __int8 v30; // [rsp+Fh] [rbp-51h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_35;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = *(__int64 **)(a1 + 8);
  v31 = v5 + 176;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F875EC )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_32;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F875EC);
  v10 = *(__int64 **)(a1 + 8);
  v11 = v9 + 176;
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F86530 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_33;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F86530);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD **)(v14 + 176);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F8D474 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_34;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F8D474);
  v32 = v11;
  v20 = *(_BYTE *)(a1 + 169);
  v30 = 0;
  v21 = *(_QWORD *)(v19 + 176);
  v33 = a2 + 72;
  v22 = *(_QWORD *)(a2 + 80);
  if ( v22 != a2 + 72 )
  {
    do
    {
      v23 = v22;
      v24 = 0;
      v25 = v21;
      v26 = v23;
      do
      {
        v27 = v26 - 24;
        if ( !v26 )
          v27 = 0;
        v28 = sub_2978BA0(v27, v31, v32, v16, v25, v20);
        v26 = *(_QWORD *)(v26 + 8);
        v24 |= v28;
      }
      while ( v26 != v33 );
      v21 = v25;
      if ( !v24 )
        break;
      v30 = v24;
      v22 = *(_QWORD *)(a2 + 80);
    }
    while ( v22 != v33 );
  }
  return v30;
}
