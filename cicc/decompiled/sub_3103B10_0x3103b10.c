// Function: sub_3103B10
// Address: 0x3103b10
//
unsigned __int64 __fastcall sub_3103B10(_BYTE *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int64 result; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rax

  if ( !a3 )
    return 0;
  v3 = *(_QWORD *)(a3 + 40);
  if ( *(_QWORD *)(v3 + 56) != a3 + 24 )
  {
    a3 = *(_QWORD *)(a3 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    result = a3 - 24;
    if ( a3 )
      return result;
  }
  if ( !*a1 )
    return 0;
  v5 = sub_31037C0((__int64)a1, v3, a3);
  if ( v5 && (v6 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    return v6 - 24;
  else
    return 0;
}
