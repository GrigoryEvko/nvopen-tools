// Function: sub_8C6530
// Address: 0x8c6530
//
__int64 __fastcall sub_8C6530(unsigned __int8 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 result; // rax
  __int64 v5; // rax
  _QWORD *v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // rax
  unsigned int v9; // eax
  char v10; // al
  _BYTE *v11; // rax

  v2 = *(_QWORD *)(a2 + 40);
  if ( (*(_BYTE *)(a2 + 89) & 4) != 0 || (v3 = 0, v2) && *(_BYTE *)(v2 + 28) == 3 )
  {
    v5 = *(_QWORD *)(v2 + 32);
    v3 = 1;
    v6 = *(_QWORD **)(v5 + 32);
    if ( v6 )
      v3 = *v6 == v5;
  }
  if ( (*(_BYTE *)(a2 - 8) & 2) == 0 )
    v3 += 2;
  if ( a1 <= 0x1Cu )
  {
    if ( a1 > 1u )
    {
      switch ( a1 )
      {
        case 2u:
        case 8u:
        case 0x1Cu:
          return v3;
        case 6u:
          if ( (unsigned int)sub_8C6470(a2) )
          {
            if ( (*(_BYTE *)(a2 + 141) & 0x20) == 0 )
              v3 += 32;
            v3 += 16;
          }
          if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) <= 2u && (*(_BYTE *)(a2 + 178) & 1) != 0 )
            v3 += 8;
          return v3;
        case 7u:
          if ( !*(_BYTE *)(a2 + 136) )
          {
            if ( *(_BYTE *)(a2 + 177) )
              v3 += 4;
            v7 = v3 + 16;
            v3 += 48;
            if ( (*(_BYTE *)(a2 + 168) & 8) != 0 )
              v3 = v7;
          }
          if ( *(char *)(a2 + 170) < 0 )
            goto LABEL_29;
          return v3;
        case 0xBu:
          v8 = *(_QWORD *)(a2 + 152);
          if ( *(_BYTE *)(v8 + 140) != 7 || (v11 = *(_BYTE **)(*(_QWORD *)(v8 + 168) + 56LL)) == 0 || (*v11 & 2) == 0 )
            v3 += 4;
          if ( *(_DWORD *)(a2 + 160) )
          {
            v9 = v3 + 16;
            v3 += 48;
            if ( (*(_BYTE *)(a2 + 200) & 0x20) != 0 )
              v3 = v9;
          }
          v10 = *(_BYTE *)(a2 + 195);
          if ( (v10 & 2) == 0
            && ((v10 & 8) == 0 || *(_BYTE *)(a2 + 174) == 7 || (*(_BYTE *)(**(_QWORD **)(a2 + 248) + 81LL) & 2) == 0) )
          {
            return v3;
          }
LABEL_29:
          result = v3 + 8;
          break;
        default:
          goto LABEL_43;
      }
      return result;
    }
LABEL_43:
    sub_721090();
  }
  if ( a1 != 59 )
    goto LABEL_43;
  if ( (*(_BYTE *)(*(_QWORD *)a2 + 81LL) & 2) != 0 )
    v3 += 16;
  return v3;
}
