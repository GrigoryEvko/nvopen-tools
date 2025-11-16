// Function: sub_25DD3F0
// Address: 0x25dd3f0
//
bool __fastcall sub_25DD3F0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // rdx

  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    return 1;
  if ( *(_BYTE *)(a2 + 28) )
  {
    v2 = *(_QWORD **)(a2 + 8);
    v3 = &v2[*(unsigned int *)(a2 + 20)];
    if ( v2 != v3 )
    {
      while ( a1 != *v2 )
      {
        if ( v3 == ++v2 )
          goto LABEL_11;
      }
      return 1;
    }
  }
  else if ( sub_C8CA60(a2, a1) )
  {
    return 1;
  }
LABEL_11:
  if ( !*(_BYTE *)(a2 + 92) )
    return sub_C8CA60(a2 + 64, a1) != 0;
  v5 = *(_QWORD **)(a2 + 72);
  v6 = &v5[*(unsigned int *)(a2 + 84)];
  if ( v5 == v6 )
    return 0;
  while ( a1 != *v5 )
  {
    if ( v6 == ++v5 )
      return 0;
  }
  return 1;
}
