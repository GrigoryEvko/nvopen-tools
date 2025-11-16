// Function: sub_1E9B6A0
// Address: 0x1e9b6a0
//
__int64 __fastcall sub_1E9B6A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  unsigned int v6; // eax
  bool v7; // cf
  __int64 v8; // rbx
  int v9; // eax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx

  v2 = sub_1636880(a1, *(_QWORD *)a2);
  if ( (_BYTE)v2 )
    return 0;
  v3 = v2;
  *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 40);
  v4 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v5 = 0;
  if ( v4 != sub_1D00B00 )
    v5 = v4();
  *(_QWORD *)(a1 + 240) = v5;
  v6 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 40LL) - 34;
  v7 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 40LL) == 34;
  *(_BYTE *)(a1 + 256) = v6 <= 1;
  if ( v7 || v6 == 1 )
  {
    v11 = *(__int64 **)(a1 + 8);
    v12 = *v11;
    v13 = v11[1];
    if ( v12 == v13 )
LABEL_14:
      BUG();
    while ( *(_UNKNOWN **)v12 != &unk_4FC62EC )
    {
      v12 += 16;
      if ( v13 == v12 )
        goto LABEL_14;
    }
    *(_QWORD *)(a1 + 248) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
                              *(_QWORD *)(v12 + 8),
                              &unk_4FC62EC);
  }
  v8 = *(_QWORD *)(a2 + 328);
  if ( v8 == a2 + 320 )
    return 0;
  do
  {
    v9 = sub_1E99E80(a1, v8);
    v8 = *(_QWORD *)(v8 + 8);
    v3 |= v9;
  }
  while ( a2 + 320 != v8 );
  return v3;
}
