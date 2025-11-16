// Function: sub_34B9CE0
// Address: 0x34b9ce0
//
__int64 __fastcall sub_34B9CE0(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdx
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 40);
  v2 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == v1 + 48 )
    goto LABEL_10;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_10:
    BUG();
  v3 = 0;
  if ( *(_BYTE *)(v2 - 24) == 30 && (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) != 0 )
  {
    v4 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) - 24);
    if ( v4 )
    {
      v5 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      LOBYTE(v3) = v4 == v5;
      LOBYTE(v5) = v5 != 0;
      v3 &= v5;
    }
  }
  return v3;
}
