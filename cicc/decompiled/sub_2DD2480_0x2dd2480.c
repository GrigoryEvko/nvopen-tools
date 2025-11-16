// Function: sub_2DD2480
// Address: 0x2dd2480
//
__int64 __fastcall sub_2DD2480(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax

  if ( (*(_BYTE *)(a2 + 3) & 0x40) == 0 )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_501DA08 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_8;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_501DA08);
  sub_2DD15F0(v6, a2);
  return sub_2DD1EE0(a2, a2);
}
