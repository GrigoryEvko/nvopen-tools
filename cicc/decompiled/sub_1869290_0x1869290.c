// Function: sub_1869290
// Address: 0x1869290
//
__int64 __fastcall sub_1869290(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax

  if ( (unsigned __int8)sub_1636800(a1, a2) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F9B6E8 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_8;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9B6E8);
  return sub_18691B0((__int64)a2, v6 + 360);
}
