// Function: sub_1D23510
// Address: 0x1d23510
//
__int64 __fastcall sub_1D23510(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int16 v3; // dx

  v1 = *(_QWORD *)(a1 + 32);
  v2 = v1 + 40LL * *(unsigned int *)(a1 + 56);
  if ( v1 == v2 )
    return 1;
  while ( 1 )
  {
    v3 = *(_WORD *)(*(_QWORD *)v1 + 24LL);
    if ( (unsigned __int16)(v3 - 10) > 1u && v3 != 48 )
      break;
    v1 += 40;
    if ( v2 == v1 )
      return 1;
  }
  return 0;
}
