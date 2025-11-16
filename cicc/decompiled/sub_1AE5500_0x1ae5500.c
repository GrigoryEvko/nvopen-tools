// Function: sub_1AE5500
// Address: 0x1ae5500
//
__int64 __fastcall sub_1AE5500(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // r12
  unsigned int v15; // r15d
  __int64 v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rdi
  int v19; // eax
  __int64 *i; // [rsp+8h] [rbp-38h]

  v1 = (__int64 *)a1[1];
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9920C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_20;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9920C);
  v6 = (__int64 *)a1[1];
  a1[21] = v5 + 160;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F9E06C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_19;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9E06C);
  v10 = a1[1];
  a1[20] = v9 + 160;
  v11 = sub_160F9A0(v10, (__int64)&unk_4F9A488, 1u);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F9A488)) != 0 )
    v13 = *(_QWORD *)(v12 + 160);
  else
    v13 = 0;
  v14 = a1[21];
  a1[22] = v13;
  v15 = 0;
  v16 = a1[20];
  v17 = *(__int64 **)(v14 + 32);
  for ( i = *(__int64 **)(v14 + 40); i != v17; v15 |= v19 )
  {
    v18 = *v17++;
    LOBYTE(v19) = sub_1AE5120(v18, v16, v14, v13);
  }
  return v15;
}
