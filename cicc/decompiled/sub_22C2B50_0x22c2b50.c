// Function: sub_22C2B50
// Address: 0x22c2b50
//
__int64 __fastcall sub_22C2B50(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8662C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8662C);
  *(_QWORD *)(a1 + 176) = sub_CFFAC0(v5, a2);
  v6 = sub_22C1580(a1 + 176);
  v7 = v6;
  if ( v6 )
  {
    sub_22C26A0(v6);
    sub_22BE510(v7 + 32);
  }
  return 0;
}
