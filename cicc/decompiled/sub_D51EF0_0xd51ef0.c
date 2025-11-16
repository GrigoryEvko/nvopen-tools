// Function: sub_D51EF0
// Address: 0xd51ef0
//
__int64 __fastcall sub_D51EF0(_QWORD *a1, __int64 a2)
{
  void (*v2)(void); // rax
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax

  v2 = *(void (**)(void))(*a1 + 96LL);
  if ( (char *)v2 == (char *)sub_D4CEA0 )
    sub_D4CC10((__int64)(a1 + 22), a2);
  else
    v2();
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_9:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F8144C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_9;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F8144C);
  sub_D50CB0((__int64)(a1 + 22), v6 + 176);
  return 0;
}
