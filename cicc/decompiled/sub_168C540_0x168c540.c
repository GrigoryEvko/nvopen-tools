// Function: sub_168C540
// Address: 0x168c540
//
__int64 __fastcall sub_168C540(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v5; // rdx

  v3 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a3 & 4) != 0 )
    return sub_16E7EE0(a2, *(const char **)v3, *(_QWORD *)(v3 + 8));
  if ( (*(_BYTE *)(v3 + 33) & 3) == 1 )
  {
    v5 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 5 )
    {
      sub_16E7EE0(a2, "__imp_", 6);
    }
    else
    {
      *(_DWORD *)v5 = 1835622239;
      *(_WORD *)(v5 + 4) = 24432;
      *(_QWORD *)(a2 + 24) += 6LL;
    }
  }
  return sub_38B9BB0(a1 + 136, a2, v3, 0);
}
