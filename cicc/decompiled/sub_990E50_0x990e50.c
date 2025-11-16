// Function: sub_990E50
// Address: 0x990e50
//
__int64 __fastcall sub_990E50(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v4; // r8
  __int64 v7; // rdx
  unsigned __int8 *v8; // rcx
  int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // r8d
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v4 = *(_QWORD *)(a1 - 8);
  v7 = 0;
  while ( 2 )
  {
    v8 = *(unsigned __int8 **)(v4 + 32 * v7);
    v9 = *v8;
    if ( (unsigned __int8)v9 <= 0x1Cu )
      goto LABEL_6;
    v10 = v9 - 42;
    if ( v10 > 0x11 )
      goto LABEL_6;
    switch ( v10 )
    {
      case 0u:
      case 2u:
      case 4u:
      case 5u:
      case 6u:
      case 9u:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0xFu:
      case 0x10u:
        v13 = *((_QWORD *)v8 - 4);
        v14 = *((_QWORD *)v8 - 8);
        if ( !v13 )
        {
          if ( a1 == v14 )
          {
            v14 = 0;
            goto LABEL_14;
          }
LABEL_6:
          if ( v7 == 1 )
            return 0;
          v7 = 1;
          continue;
        }
        if ( a1 != v14 )
        {
          if ( a1 == v13 )
            goto LABEL_14;
          goto LABEL_6;
        }
        v14 = *((_QWORD *)v8 - 4);
LABEL_14:
        v15 = *(_QWORD *)(v4 + 32LL * ((unsigned int)v7 ^ 1));
        *a2 = v8;
        v11 = 1;
        *a3 = v15;
        *a4 = v14;
        return v11;
      default:
        goto LABEL_6;
    }
  }
}
