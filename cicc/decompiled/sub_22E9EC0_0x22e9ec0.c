// Function: sub_22E9EC0
// Address: 0x22e9ec0
//
__int64 __fastcall sub_22E9EC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_6:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FDBD0C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_6;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FDBD0C);
  sub_22E9250(a2, v5 + 176, *(_BYTE **)(a1 + 176), *(_QWORD *)(a1 + 184), 0);
  return 0;
}
