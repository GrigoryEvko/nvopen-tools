// Function: sub_1C51E40
// Address: 0x1c51e40
//
char __fastcall sub_1C51E40(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rbx
  __int64 v3; // rdx
  char result; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r12

  v1 = *(unsigned __int16 *)(a1 + 24);
  v2 = a1;
  if ( (_WORD)v1 == 10 )
  {
LABEL_2:
    v3 = *(_QWORD *)(v2 - 8);
    result = 0;
    if ( *(_BYTE *)(v3 + 16) == 78 )
    {
      v5 = *(_QWORD *)(v3 - 24);
      result = 1;
      if ( !*(_BYTE *)(v5 + 16) )
        return (*(_BYTE *)(v5 + 33) & 0x20) == 0;
    }
    return result;
  }
  while ( 1 )
  {
    if ( !(_WORD)v1 )
      return 0;
    if ( (unsigned __int16)(v1 - 1) <= 2u )
    {
      v2 = *(_QWORD *)(v2 + 32);
      goto LABEL_6;
    }
    if ( (_WORD)v1 != 6 )
      break;
    if ( (unsigned __int8)sub_1C51E40(*(_QWORD *)(v2 + 32)) )
      return 1;
    v2 = *(_QWORD *)(v2 + 40);
LABEL_6:
    v1 = *(unsigned __int16 *)(v2 + 24);
    if ( (_WORD)v1 == 10 )
      goto LABEL_2;
  }
  result = (unsigned __int16)(v1 - 7) <= 2u || (unsigned int)(v1 - 4) <= 1;
  if ( result )
  {
    v6 = *(_QWORD *)(v2 + 40);
    if ( !(_DWORD)v6 )
      return 0;
    v7 = *(_QWORD **)(v2 + 32);
    v8 = (__int64)&v7[(unsigned int)(v6 - 1) + 1];
    while ( 1 )
    {
      result = sub_1C51E40(*v7);
      if ( result )
        break;
      if ( ++v7 == (_QWORD *)v8 )
        return result;
    }
    return 1;
  }
  return result;
}
