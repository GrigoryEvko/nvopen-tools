// Function: sub_19CECC0
// Address: 0x19cecc0
//
__int64 __fastcall sub_19CECC0(__int64 *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax

  v1 = *(__int64 **)(*a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9D764 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_6;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9D764);
  return sub_14CF090(v4, a1[1]);
}
