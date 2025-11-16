// Function: sub_18ED5E0
// Address: 0x18ed5e0
//
__int64 __fastcall sub_18ED5E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v2 = *(_QWORD *)(a2 + 80);
  if ( v2 )
    v2 -= 24;
  v3 = 0;
  if ( byte_4FAE240 )
  {
    v15 = *(__int64 **)(a1 + 8);
    v16 = *v15;
    v17 = v15[1];
    if ( v17 == v16 )
      goto LABEL_20;
    while ( *(_UNKNOWN **)v16 != &unk_4F97E48 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_20;
    }
    v3 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
           *(_QWORD *)(v16 + 8),
           &unk_4F97E48)
       + 160;
  }
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
    goto LABEL_20;
  while ( *(_UNKNOWN **)v5 != &unk_4F9E06C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_20;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F9E06C);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7 + 160;
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F9D3C0 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_20;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F9D3C0);
  v13 = sub_14A4050(v12, a2);
  return sub_18ED570(a1 + 160, a2, v13, v9, v3, v2);
}
