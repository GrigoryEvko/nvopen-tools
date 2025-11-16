// Function: sub_1601F00
// Address: 0x1601f00
//
__int64 __fastcall sub_1601F00(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // rax
  _BYTE *v11; // rdi
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // cl
  char v16; // si

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    v9 = 0;
    goto LABEL_7;
  }
  v3 = sub_1648A40(a1);
  v5 = v3 + v4;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v5 >> 4) )
LABEL_42:
      BUG();
LABEL_18:
    v9 = 0;
    v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    goto LABEL_7;
  }
  if ( !(unsigned int)((v5 - sub_1648A40(a1)) >> 4) )
    goto LABEL_18;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_42;
  v6 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v7 = sub_1648A40(a1);
  v9 = *(_DWORD *)(v7 + v8 - 4) - v6;
  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
LABEL_7:
  v10 = *(_QWORD *)(a1 + 24 * ((unsigned int)(v2 - 3 - v9) - v1));
  if ( *(_BYTE *)(v10 + 16) != 19 )
    BUG();
  v11 = *(_BYTE **)(v10 + 24);
  result = 0;
  if ( v11 && !*v11 )
  {
    v13 = sub_161E970(v11);
    if ( v14 != 13 )
    {
      switch ( v14 )
      {
        case 15LL:
          if ( *(_QWORD *)v13 == 0x6F742E646E756F72LL
            && *(_DWORD *)(v13 + 8) == 1918985582
            && *(_WORD *)(v13 + 12) == 29541
            && *(_BYTE *)(v13 + 14) == 116 )
          {
            return 2;
          }
          break;
        case 14LL:
          if ( *(_QWORD *)v13 == 0x6F642E646E756F72LL
            && *(_DWORD *)(v13 + 8) == 1635217015
            && *(_WORD *)(v13 + 12) == 25714 )
          {
            return 3;
          }
          break;
        case 12LL:
          if ( *(_QWORD *)v13 == 0x70752E646E756F72LL && *(_DWORD *)(v13 + 8) == 1685217655 )
            return 4;
          return 0;
        default:
          v16 = 1;
          v15 = 0;
          goto LABEL_27;
      }
      return 0;
    }
    if ( *(_QWORD *)v13 != 0x79642E646E756F72LL
      || *(_DWORD *)(v13 + 8) != 1768776046
      || (v16 = 0, v15 = 1, *(_BYTE *)(v13 + 12) != 99) )
    {
      v15 = 0;
      v16 = 1;
    }
LABEL_27:
    if ( v14 == 16 && v16 )
    {
      return (*(_QWORD *)(v13 + 8) ^ 0x6F72657A64726177LL | *(_QWORD *)v13 ^ 0x6F742E646E756F72LL) == 0 ? 5 : 0;
    }
    else
    {
      result = 1;
      if ( !v15 )
        return 0;
    }
  }
  return result;
}
