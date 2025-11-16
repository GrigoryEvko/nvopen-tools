// Function: sub_1983870
// Address: 0x1983870
//
__int64 __fastcall sub_1983870(__int64 a1, __int64 a2, double a3, __m128i a4)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD v14[12]; // [rsp+0h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_15:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9A488 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_15;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9A488);
  v9 = *(__int64 **)(a1 + 8);
  v10 = *(_QWORD *)(v8 + 160);
  v11 = *v9;
  v12 = v9[1];
  if ( v11 == v12 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F98724 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_16;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F98724);
  v14[0] = v10;
  v14[1] = v13 + 160;
  return sub_1981CC0((__int64)v14, a2, a3, a4);
}
