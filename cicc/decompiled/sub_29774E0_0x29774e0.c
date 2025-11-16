// Function: sub_29774E0
// Address: 0x29774e0
//
__int64 __fastcall sub_29774E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int64 *v6; // r13
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 **v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx

  if ( (unsigned __int8)sub_BB98D0((_QWORD *)a1, a2)
    || *(_QWORD *)(a1 + 216) && !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(a1 + 224))(a1 + 200, a2) )
  {
    return 0;
  }
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_25;
  while ( *(_UNKNOWN **)v3 != &unk_4F8662C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_25;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8662C);
  *(_QWORD *)(a1 + 192) = sub_CFFAC0(v5, a2);
  if ( LOBYTE(qword_4F8D3A0[17]) )
  {
    v19 = *(__int64 **)(a1 + 8);
    v20 = *v19;
    v21 = v19[1];
    if ( v20 == v21 )
      goto LABEL_25;
    while ( *(_UNKNOWN **)v20 != &unk_4F8144C )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_25;
    }
    v6 = (unsigned __int64 *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
                                *(_QWORD *)(v20 + 8),
                                &unk_4F8144C)
                            + 176);
  }
  else
  {
    v6 = 0;
  }
  v7 = *(__int64 **)(a1 + 8);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
    goto LABEL_25;
  while ( *(_UNKNOWN **)v8 != &unk_4F89C28 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_25;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F89C28);
  v11 = sub_DFED00(v10, a2);
  v12 = *(__int64 **)(a1 + 8);
  v13 = (__int64 **)v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F8FC84 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_25;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F8FC84);
  return sub_29760C0(a2, v13, v6, v16 + 184, a1 + 176, v17);
}
