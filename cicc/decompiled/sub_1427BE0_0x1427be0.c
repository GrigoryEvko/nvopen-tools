// Function: sub_1427BE0
// Address: 0x1427be0
//
__int64 __fastcall sub_1427BE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r12

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_18:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_18;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 160;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F96DB4 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_17;
  }
  v10 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
                      *(_QWORD *)(v8 + 8),
                      &unk_4F96DB4)
                  + 160);
  v11 = sub_22077B0(344);
  v12 = v11;
  if ( v11 )
    sub_1427AF0(v11, a2, v10, v7);
  v13 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v12;
  if ( v13 )
  {
    sub_1421ED0(v13);
    j_j___libc_free_0(v13, 344);
  }
  return 0;
}
