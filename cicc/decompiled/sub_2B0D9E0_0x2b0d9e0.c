// Function: sub_2B0D9E0
// Address: 0x2b0d9e0
//
__int64 __fastcall sub_2B0D9E0(unsigned __int64 a1)
{
  unsigned int v1; // r8d
  unsigned int v3; // ecx
  unsigned int v4; // esi
  _QWORD *v5; // rax
  int v6; // ecx

  v1 = a1 & 1;
  if ( (a1 & 1) != 0 )
  {
    LOBYTE(v1) = (~(-1LL << (a1 >> 58)) & (a1 >> 1)) == (1LL << (a1 >> 58)) - 1;
    return v1;
  }
  v3 = *(_DWORD *)(a1 + 64);
  v4 = v3 >> 6;
  if ( v3 >> 6 )
  {
    v5 = *(_QWORD **)a1;
    while ( *v5 == -1 )
    {
      if ( ++v5 == (_QWORD *)(*(_QWORD *)a1 + 8LL * (v4 - 1) + 8) )
        goto LABEL_8;
    }
    return v1;
  }
LABEL_8:
  v1 = 1;
  v6 = v3 & 0x3F;
  if ( !v6 )
    return v1;
  LOBYTE(v1) = *(_QWORD *)(*(_QWORD *)a1 + 8LL * v4) == (1LL << v6) - 1;
  return v1;
}
