// Function: sub_867060
// Address: 0x867060
//
_BOOL8 __fastcall sub_867060(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _BOOL8 v4; // r8
  __int64 v5; // rax

  LODWORD(v4) = 0;
  if ( !qword_4F04C18 )
    return v4;
  if ( *((_BYTE *)qword_4F04C18 + 42) )
    return v4;
  v5 = qword_4F04C18[1];
  v4 = *(_QWORD *)(v5 + 24) != 0;
  if ( *(_QWORD *)(v5 + 24) )
    return v4;
  if ( dword_4F04C44 == -1 && (a3 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0) )
    return v4;
  else
    return *(_QWORD *)(sub_85B130(a1, a2, a3, a4, v4) + 664) != 0;
}
