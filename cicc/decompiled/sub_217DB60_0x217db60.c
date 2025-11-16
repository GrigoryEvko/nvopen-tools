// Function: sub_217DB60
// Address: 0x217db60
//
__int64 __fastcall sub_217DB60(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx

  v1 = *(_DWORD *)(a1 + 40);
  if ( !v1 )
    return v1;
  v2 = v1 - 1;
  v3 = *(_QWORD *)(a1 + 32);
  v1 = 0;
  v4 = v3 + 40 * v2 + 40;
  while ( 1 )
  {
    if ( *(_BYTE *)v3 || (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
      goto LABEL_7;
    if ( v1 )
      return 0;
    v1 = *(_DWORD *)(v3 + 8);
LABEL_7:
    v3 += 40;
    if ( v4 == v3 )
      return v1;
  }
}
