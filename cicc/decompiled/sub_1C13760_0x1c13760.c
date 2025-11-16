// Function: sub_1C13760
// Address: 0x1c13760
//
__int64 __fastcall sub_1C13760(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdi

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F99CCC )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_12;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F99CCC)
     + 160;
  v5 = sub_22077B0(88);
  v6 = v5;
  if ( v5 )
    sub_1C13660(v5, v4);
  v7 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v6;
  if ( v7 )
  {
    v8 = *(_QWORD *)(v7 + 56);
    *(_QWORD *)v7 = &unk_49F7548;
    if ( v8 )
      j_j___libc_free_0(v8, *(_QWORD *)(v7 + 72) - v8);
    sub_1C12880(*(_QWORD **)(v7 + 24));
    j_j___libc_free_0(v7, 88);
  }
  return 0;
}
