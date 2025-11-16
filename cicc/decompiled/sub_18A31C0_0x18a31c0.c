// Function: sub_18A31C0
// Address: 0x18a31c0
//
__int64 __fastcall sub_18A31C0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax

  if ( (unsigned __int8)sub_1636800(a1, a2) )
    return 0;
  v3 = sub_1632FA0((__int64)a2);
  v4 = *(__int64 **)(a1 + 8);
  v5 = v3;
  v6 = *v4;
  v7 = v4[1];
  if ( v6 == v7 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9B6E8 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_8;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9B6E8);
  return sub_1A18770(a2, v5, v8 + 360);
}
