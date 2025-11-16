// Function: sub_2E05E30
// Address: 0x2e05e30
//
__int64 __fastcall sub_2E05E30(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 *v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // r15

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_10:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EACC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_10;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501EACC)
     + 200;
  v6 = (__int64 *)sub_22077B0(8u);
  v7 = v6;
  if ( v6 )
    sub_2DF8860(v6);
  v8 = *(__int64 **)(a1 + 200);
  *(_QWORD *)(a1 + 200) = v7;
  if ( v8 )
  {
    sub_2DFA680(v8);
    j_j___libc_free_0((unsigned __int64)v8);
    v7 = *(__int64 **)(a1 + 200);
  }
  sub_2E05B00(v7, a2, v5);
  return 0;
}
