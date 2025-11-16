// Function: sub_1DD6160
// Address: 0x1dd6160
//
unsigned __int64 __fastcall sub_1DD6160(__int64 a1)
{
  __int64 v1; // rsi
  unsigned __int64 v2; // rcx
  unsigned __int64 v3; // rdi
  unsigned __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 32);
  v2 = a1 + 24;
  v3 = v2;
  if ( v1 == v2 )
    return v2;
  while ( 1 )
  {
    v3 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
    result = v3;
    if ( (unsigned __int16)(**(_WORD **)(v3 + 16) - 12) > 1u && (*(_BYTE *)(v3 + 46) & 4) == 0 )
      break;
    if ( v1 == v3 )
      return v2;
  }
  return result;
}
