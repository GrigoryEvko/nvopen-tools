// Function: sub_35B8B40
// Address: 0x35b8b40
//
__int64 __fastcall sub_35B8B40(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  v2 = *(_BYTE *)(a1 + 332) == 0;
  *(_QWORD *)a1 = off_4A3A2E0;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 312));
  v3 = *(_QWORD *)(a1 + 272);
  while ( v3 )
  {
    sub_35B8760(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  v5 = *(_QWORD *)(a1 + 224);
  while ( v5 )
  {
    sub_35B8760(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
