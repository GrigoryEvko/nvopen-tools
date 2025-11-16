// Function: sub_22AB330
// Address: 0x22ab330
//
__int64 __fastcall sub_22AB330(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdx
  unsigned __int64 v4; // r13
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdi

  v3 = (_QWORD *)sub_22077B0(0x60u);
  if ( v3 )
  {
    memset(v3, 0, 0x60u);
    *v3 = v3 + 2;
    v3[1] = 0x100000000LL;
  }
  v4 = a1[22];
  a1[22] = v3;
  if ( v4 )
  {
    sub_C7D6A0(*(_QWORD *)(v4 + 56), 16LL * *(unsigned int *)(v4 + 72), 8);
    if ( *(_QWORD *)v4 != v4 + 16 )
      _libc_free(*(_QWORD *)v4);
    j_j___libc_free_0(v4);
  }
  v5 = (__int64 *)a1[1];
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4FDB68C )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_12;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4FDB68C);
  v9 = a1[22];
  a1[23] = v8 + 176;
  sub_22AA8A0(v9, a2, v8 + 176);
  return 0;
}
