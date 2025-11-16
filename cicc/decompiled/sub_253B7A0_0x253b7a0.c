// Function: sub_253B7A0
// Address: 0x253b7a0
//
__int64 __fastcall sub_253B7A0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rsi
  __int64 v4; // rdx
  _QWORD *v5; // rcx
  unsigned int v6; // r8d
  _QWORD *v8; // rax

  v1 = *(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 28) )
    v2 = *(unsigned int *)(a1 + 20);
  else
    v2 = *(unsigned int *)(a1 + 16);
  v3 = &v1[v2];
  if ( v1 == v3 )
    return 0;
  while ( 1 )
  {
    v4 = *v1;
    v5 = v1;
    if ( *v1 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v3 == ++v1 )
      return 0;
  }
  if ( v3 == v1 )
  {
    return 0;
  }
  else
  {
    v6 = 0;
    do
    {
      v8 = v5 + 1;
      v6 += ((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9);
      if ( v5 + 1 == v3 )
        break;
      while ( 1 )
      {
        v4 = *v8;
        v5 = v8;
        if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v3 == ++v8 )
          return v6;
      }
    }
    while ( v3 != v8 );
  }
  return v6;
}
