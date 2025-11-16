// Function: sub_AF2D10
// Address: 0xaf2d10
//
__int64 __fastcall sub_AF2D10(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) == 0 )
  {
    v2 = *(_QWORD *)(a1 - 16 - 8LL * ((v1 >> 2) & 0xF) + 32);
    if ( v2 )
      goto LABEL_3;
    return 0;
  }
  v2 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 32LL);
  if ( !v2 )
    return 0;
LABEL_3:
  v3 = *(_QWORD *)(v2 + 136);
  result = 0;
  if ( v3 && *(_BYTE *)v3 == 17 )
  {
    result = *(_QWORD *)(v3 + 24);
    if ( *(_DWORD *)(v3 + 32) > 0x40u )
      return *(_QWORD *)result;
  }
  return result;
}
