// Function: sub_1E5E370
// Address: 0x1e5e370
//
__int64 __fastcall sub_1E5E370(_QWORD *a1, __int64 *a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 (*v12)(void); // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD **v15; // rbx
  _QWORD **v16; // r13
  _QWORD *v17; // rsi
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_1636880((__int64)a1, *a2) )
    return 0;
  if ( !byte_4FC71C0 )
    return 0;
  v18[0] = *(_QWORD *)(*a2 + 112);
  if ( (unsigned __int8)sub_1560260(v18, -1, 34) )
  {
    if ( !dword_4FC7050 )
      return 0;
  }
  v3 = (__int64 *)a1[1];
  a1[29] = a2;
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4FC6A0C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_23;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4FC6A0C);
  v7 = (__int64 *)a1[1];
  a1[30] = v6;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_22:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC62EC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_22;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4FC62EC);
  v11 = a1[29];
  a1[31] = v10;
  v12 = *(__int64 (**)(void))(**(_QWORD **)(v11 + 16) + 40LL);
  v13 = 0;
  if ( v12 != sub_1D00B00 )
    v13 = v12();
  a1[33] = v13;
  sub_1ED7320(a1 + 34);
  v14 = a1[30];
  v15 = *(_QWORD ***)(v14 + 264);
  v16 = *(_QWORD ***)(v14 + 272);
  if ( v15 == v16 )
    return 0;
  do
  {
    v17 = *v15++;
    sub_1E5DDB0((__int64)a1, v17);
  }
  while ( v16 != v15 );
  return 0;
}
