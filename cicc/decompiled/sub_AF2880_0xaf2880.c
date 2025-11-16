// Function: sub_AF2880
// Address: 0xaf2880
//
unsigned __int64 __fastcall sub_AF2880(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rax
  unsigned __int8 v3; // dl

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 16LL);
    if ( !v2 )
      return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 - 8LL * ((v1 >> 2) & 0xF));
    if ( !v2 )
      return 0;
  }
  v3 = *(_BYTE *)v2;
  if ( *(_BYTE *)v2 == 1 )
    return *(_QWORD *)(v2 + 136) & 0xFFFFFFFFFFFFFFF9LL;
  if ( (unsigned int)v3 - 25 <= 1 )
    return v2 & 0xFFFFFFFFFFFFFFF9LL | 2;
  if ( v3 != 7 )
    return 0;
  return v2 & 0xFFFFFFFFFFFFFFF9LL | 4;
}
