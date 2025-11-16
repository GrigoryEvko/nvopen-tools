// Function: sub_13831E0
// Address: 0x13831e0
//
void __fastcall sub_13831E0(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9B6E8 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_10;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9B6E8)
     + 360;
  v5 = sub_22077B0(56);
  v6 = v5;
  if ( v5 )
    sub_1383040(v5, v4);
  v7 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v6;
  if ( v7 )
  {
    sub_1383070(v7);
    j_j___libc_free_0(v7, 56);
  }
}
