// Function: sub_17C4A30
// Address: 0x17c4a30
//
__int64 __fastcall sub_17C4A30(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r9

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F99CCD )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F99CCD);
  if ( byte_4FA3280 )
    return 0;
  else
    return sub_17C2DB0(
             a2,
             *(_QWORD **)(v5 + 160),
             byte_4FA2FE0 | *(_BYTE *)(a1 + 153),
             byte_4FA2F00 | *(_BYTE *)(a1 + 154),
             0,
             v6);
}
