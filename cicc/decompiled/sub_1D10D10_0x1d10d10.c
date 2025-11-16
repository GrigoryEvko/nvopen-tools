// Function: sub_1D10D10
// Address: 0x1d10d10
//
void __fastcall sub_1D10D10(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 i; // r12

  v1 = *(_QWORD *)(a1 + 624);
  v2 = *(_QWORD *)(v1 + 200);
  for ( i = v1 + 192; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    while ( !v2
         || *(__int16 *)(v2 + 16) >= 0
         || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) + ((__int64)~*(__int16 *)(v2 + 16) << 6) + 10) & 1) == 0 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( i == v2 )
        return;
    }
    sub_1D0FB70(a1, v2 - 8);
  }
}
