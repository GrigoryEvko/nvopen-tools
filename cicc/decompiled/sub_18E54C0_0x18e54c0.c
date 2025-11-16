// Function: sub_18E54C0
// Address: 0x18e54c0
//
__int64 __fastcall sub_18E54C0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v6 != &unk_4F9D764 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_16;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9D764);
  v9 = sub_14CF090(v8, a2);
  v10 = *(__int64 **)(a1 + 8);
  v11 = v9;
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v12 != &unk_4F9A488 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_16;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9A488);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD *)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9E06C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_16;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9E06C);
  return sub_18E53D0((__int64 *)(a1 + 160), a3, a4, a2, v11, v16, v19 + 160);
}
