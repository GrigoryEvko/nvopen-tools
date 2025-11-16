// Function: sub_1B44C50
// Address: 0x1b44c50
//
__int64 *__fastcall sub_1B44C50(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r14
  __int64 v4; // r15
  int v5; // ebx
  __int64 **v6; // r12
  __int64 *v7; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // eax
  __int64 *v12; // rbx
  __int64 v13; // r14
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 27 )
  {
    v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v14 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
    v4 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
    if ( !v4 )
      goto LABEL_9;
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v4) + 16) - 25) > 9u )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_9;
    }
    v5 = 0;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v4) + 16) - 25) <= 9u )
      {
        v4 = *(_QWORD *)(v4 + 8);
        ++v5;
        if ( !v4 )
          goto LABEL_8;
      }
    }
LABEL_8:
    if ( v14 * (v5 + 1) <= 0x80 )
    {
LABEL_9:
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v6 = *(__int64 ***)(a2 - 8);
      else
        v6 = (__int64 **)(a2 - 24 * v3);
      v7 = *v6;
      if ( v7 )
        goto LABEL_12;
    }
    return 0;
  }
  if ( v2 != 26 )
    return 0;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v9 = *(_QWORD *)(a2 - 72);
  v10 = *(_QWORD *)(v9 + 8);
  if ( !v10 )
    return 0;
  if ( *(_QWORD *)(v10 + 8) )
    return 0;
  if ( *(_BYTE *)(v9 + 16) != 75 )
    return 0;
  v11 = *(unsigned __int16 *)(v9 + 18);
  BYTE1(v11) &= ~0x80u;
  if ( (unsigned int)(v11 - 32) > 1 )
    return 0;
  if ( !sub_1B42400(*(__int64 ****)(v9 - 24), *(_QWORD *)(a1 + 8)) )
    return 0;
  v7 = *(__int64 **)(v9 - 48);
  if ( !v7 )
    return 0;
LABEL_12:
  if ( *((_BYTE *)v7 + 16) == 69 )
  {
    v12 = (__int64 *)*(v7 - 3);
    v13 = *v7;
    if ( v13 == sub_15A9650(*(_QWORD *)(a1 + 8), *v12) )
      return v12;
  }
  return v7;
}
