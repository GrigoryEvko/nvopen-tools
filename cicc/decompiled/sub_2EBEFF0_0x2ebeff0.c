// Function: sub_2EBEFF0
// Address: 0x2ebeff0
//
__int64 __fastcall sub_2EBEFF0(__int64 a1, int a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdx

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  v3 = 0;
  if ( v2 )
  {
    while ( (*(_BYTE *)(v2 + 3) & 0x10) != 0 || (*(_BYTE *)(v2 + 4) & 8) != 0 )
    {
      v2 = *(_QWORD *)(v2 + 32);
      if ( !v2 )
        return 0;
    }
    v4 = *(_QWORD *)(v2 + 16);
    do
      v2 = *(_QWORD *)(v2 + 32);
    while ( v2 && ((*(_BYTE *)(v2 + 3) & 0x10) != 0 || (*(_BYTE *)(v2 + 4) & 8) != 0 || v4 == *(_QWORD *)(v2 + 16)) );
    LOBYTE(v3) = v2 == 0;
  }
  return v3;
}
