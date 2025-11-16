// Function: sub_1401A70
// Address: 0x1401a70
//
__int64 __fastcall sub_1401A70(_QWORD *a1)
{
  void (*v1)(void); // rax
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax

  v1 = *(void (**)(void))(*a1 + 96LL);
  if ( (char *)v1 == (char *)sub_13FB660 )
    sub_13FB2B0((__int64)(a1 + 20));
  else
    v1();
  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_9:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_9;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C);
  sub_1400540((__int64)(a1 + 20), v5 + 160);
  return 0;
}
