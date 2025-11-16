// Function: sub_2DB1540
// Address: 0x2db1540
//
__int64 __fastcall sub_2DB1540(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  _DWORD *v6; // rbx
  __int64 (*v7)(); // rax
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 *v13; // r14
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 **v27; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_34;
  }
  v6 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                      *(_QWORD *)(v3 + 8),
                      &unk_5027190)
                  + 256);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 16LL);
  if ( v7 == sub_23CE270 )
    BUG();
  v8 = 0;
  v9 = ((__int64 (__fastcall *)(_DWORD *, __int64))v7)(v6, a2);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v10 != sub_2C8F680 )
    v8 = ((__int64 (__fastcall *)(__int64))v10)(v9);
  v11 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F8144C)) != 0 )
  {
    v13 = (unsigned __int64 *)(v12 + 176);
    if ( !*(_DWORD *)(a1 + 172) )
    {
      v18 = 0;
      goto LABEL_16;
    }
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 172) )
    {
      v18 = 0;
      v13 = 0;
      goto LABEL_16;
    }
    v24 = *(__int64 **)(a1 + 8);
    v25 = *v24;
    v26 = v24[1];
    if ( v25 == v26 )
LABEL_33:
      BUG();
    while ( *(_UNKNOWN **)v25 != &unk_4F8144C )
    {
      v25 += 16;
      if ( v26 == v25 )
        goto LABEL_33;
    }
    v13 = (unsigned __int64 *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(
                                 *(_QWORD *)(v25 + 8),
                                 &unk_4F8144C)
                             + 176);
  }
  v14 = *(__int64 **)(a1 + 8);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F89C28 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_35;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F89C28);
  v18 = sub_DFED00(v17, a2);
LABEL_16:
  v19 = *(__int64 **)(a1 + 8);
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F8FC84 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_36;
  }
  v27 = (__int64 **)v18;
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F8FC84);
  return sub_2DB0210(*(_DWORD *)(a1 + 172), a2, v8, v13, v27, v22 + 184, v6 + 128);
}
