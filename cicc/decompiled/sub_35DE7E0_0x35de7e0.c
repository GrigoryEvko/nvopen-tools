// Function: sub_35DE7E0
// Address: 0x35de7e0
//
char __fastcall sub_35DE7E0(_DWORD *a1, unsigned __int8 *a2)
{
  int v3; // eax
  int v4; // r13d
  __int64 v5; // rdi
  char v6; // al
  unsigned int v7; // eax
  char result; // al
  __int64 v9; // rdi
  char v10; // al
  unsigned __int8 *v11; // rbx
  __int64 v12; // rdi
  char v13; // al
  unsigned int v14; // eax
  unsigned __int8 *v15; // rbx
  __int64 v16; // rdi
  unsigned __int8 *v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rax
  _QWORD v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *a2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (unsigned __int8)v3 <= 0x15u && (_BYTE)v3 != 5 || (_BYTE)v3 == 22 )
    {
LABEL_3:
      v5 = *((_QWORD *)a2 + 1);
      v6 = *(_BYTE *)(v5 + 8);
      if ( v6 == 14 || v6 == 7 )
      {
        return 1;
      }
      else
      {
LABEL_5:
        if ( v6 != 12 )
          return 0;
        v7 = *(_DWORD *)(v5 + 8) >> 8;
        return v7 != 1 && v7 <= a1[6] && (unsigned int)sub_BCB060(v5) <= *a1;
      }
    }
    else
    {
      return (_BYTE)v3 == 23;
    }
  }
  else
  {
    v4 = v3 - 29;
    switch ( *a2 )
    {
      case 0x1Eu:
      case 0x3Du:
      case 0x43u:
      case 0x54u:
      case 0x56u:
        goto LABEL_3;
      case 0x1Fu:
      case 0x20u:
      case 0x3Eu:
      case 0x3Fu:
        return 1;
      case 0x44u:
        if ( (a2[7] & 0x40) != 0 )
          v11 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v11 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v5 = *(_QWORD *)(*(_QWORD *)v11 + 8LL);
        v6 = *(_BYTE *)(v5 + 8);
        if ( v6 == 7 || v6 == 14 )
          return 1;
        goto LABEL_5;
      case 0x4Eu:
        if ( (a2[7] & 0x40) != 0 )
          v17 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v17 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        return *(_QWORD *)(*(_QWORD *)v17 + 8LL) == *((_QWORD *)a2 + 1);
      case 0x52u:
        if ( (a2[7] & 0x40) != 0 )
          v15 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v15 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v16 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
        if ( *(_BYTE *)(v16 + 8) == 14 )
          return 1;
        return *a1 == (unsigned int)sub_BCB060(v16);
      case 0x55u:
        v9 = *((_QWORD *)a2 + 1);
        v10 = *(_BYTE *)(v9 + 8);
        if ( v10 != 14 && v10 != 7 )
        {
          if ( v10 != 12 )
            return 0;
          v18 = *(_DWORD *)(v9 + 8) >> 8;
          if ( v18 == 1 || v18 > a1[6] || (unsigned int)sub_BCB060(v9) > *a1 )
            return 0;
        }
        if ( (unsigned __int8)sub_A74710((_QWORD *)a2 + 9, 0, 79) )
          return 1;
        v19 = *((_QWORD *)a2 - 4);
        if ( !v19 || *(_BYTE *)v19 || *(_QWORD *)(v19 + 24) != *((_QWORD *)a2 + 10) )
          return 0;
        v20[0] = *(_QWORD *)(v19 + 120);
        result = sub_A74710(v20, 0, 79);
        break;
      default:
        if ( (unsigned int)(v3 - 42) > 0x11 )
          return 0;
        v12 = *((_QWORD *)a2 + 1);
        v13 = *(_BYTE *)(v12 + 8);
        if ( v13 != 14 && v13 != 7 )
        {
          if ( v13 != 12 )
            return 0;
          v14 = *(_DWORD *)(v12 + 8) >> 8;
          if ( v14 == 1 || v14 > a1[6] || (unsigned int)sub_BCB060(v12) > *a1 )
            return 0;
        }
        if ( v4 == 27 || v4 == 20 )
          return 0;
        return v4 != 40 && v4 != 23;
    }
  }
  return result;
}
