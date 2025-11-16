// Function: sub_17F4120
// Address: 0x17f4120
//
__int64 __fastcall sub_17F4120(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
    goto LABEL_17;
  while ( *(_UNKNOWN **)v3 != &unk_4F97E48 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_17;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F97E48);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F99CB0 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_17;
  }
  v10 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
                      *(_QWORD *)(v8 + 8),
                      &unk_4F99CB0)
                  + 160);
  v11 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F9E06C)) != 0 )
    v13 = v12 + 160;
  else
    v13 = 0;
  if ( byte_4FA5DA0 || (unsigned __int8)sub_1560180(a2 + 112, 34) )
    return 0;
  else
    return sub_17F3F00(a2, v7 + 160, v10, v13);
}
