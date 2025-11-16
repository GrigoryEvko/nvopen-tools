// Function: sub_324C9F0
// Address: 0x324c9f0
//
__int64 __fastcall sub_324C9F0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // al
  __int64 v5; // r13
  unsigned __int8 v6; // al
  int v7; // eax
  int v8; // edx
  __int64 v9; // r8
  __int64 result; // rax
  __int16 v11; // dx
  __int64 v12; // r8
  __int64 v13; // rdx

  v4 = *(_BYTE *)(a3 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 24LL);
    if ( v5 )
      goto LABEL_3;
  }
  else
  {
    v5 = *(_QWORD *)(a3 - 16 - 8LL * ((v4 >> 2) & 0xF) + 24);
    if ( v5 )
    {
LABEL_3:
      v6 = *(_BYTE *)(v5 - 16);
      if ( (v6 & 2) != 0 )
      {
        v7 = *(_DWORD *)(v5 - 24);
        if ( !v7 )
        {
LABEL_5:
          sub_324C890(a1, a2, v5);
          goto LABEL_6;
        }
        v12 = **(_QWORD **)(v5 - 32);
        if ( !v12 )
          goto LABEL_30;
      }
      else
      {
        v11 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
        if ( ((*(_WORD *)(v5 - 16) >> 6) & 0xF) == 0 )
          goto LABEL_5;
        v12 = *(_QWORD *)(v5 - 8LL * ((v6 >> 2) & 0xF) - 16);
        if ( !v12 )
          goto LABEL_38;
      }
      sub_32495E0(a1, a2, v12, 73);
      v6 = *(_BYTE *)(v5 - 16);
      if ( (v6 & 2) != 0 )
      {
        v7 = *(_DWORD *)(v5 - 24);
LABEL_30:
        if ( v7 != 2 )
          goto LABEL_5;
        v13 = *(_QWORD *)(v5 - 32);
LABEL_32:
        if ( !*(_QWORD *)(v13 + 8) )
        {
          sub_324C890(a1, a2, v5);
          goto LABEL_10;
        }
        goto LABEL_5;
      }
      v11 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
LABEL_38:
      if ( (_BYTE)v11 != 2 )
        goto LABEL_5;
      v13 = v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
      goto LABEL_32;
    }
  }
  sub_324C890(a1, a2, 0);
LABEL_6:
  v8 = *(_DWORD *)(a1[10] + 16);
  if ( (unsigned __int16)v8 <= 0x42u )
  {
    if ( (_WORD)v8 )
    {
      switch ( (__int16)v8 )
      {
        case 1:
        case 2:
        case 12:
        case 16:
        case 29:
        case 44:
          sub_3249FA0(a1, a2, 39);
          goto LABEL_10;
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 13:
        case 14:
        case 15:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 42:
        case 43:
        case 45:
        case 46:
        case 47:
        case 48:
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
        case 56:
        case 57:
        case 61:
        case 64:
        case 65:
        case 66:
          goto LABEL_10;
        default:
          goto LABEL_18;
      }
    }
LABEL_40:
    BUG();
  }
  if ( (unsigned __int16)v8 != 45056 )
  {
    if ( (unsigned __int16)v8 <= 0xB000u )
    {
      if ( (unsigned __int16)v8 > 0x8001u || (v8 & 0x8000) != 0 )
        goto LABEL_10;
    }
    else if ( (unsigned __int16)v8 == 0xFFFF )
    {
      goto LABEL_10;
    }
LABEL_18:
    if ( (unsigned int)(unsigned __int16)v8 - 32769 > 0x7FFD )
      goto LABEL_40;
  }
LABEL_10:
  v9 = *(unsigned __int8 *)(a3 + 44);
  if ( (unsigned __int8)v9 > 1u )
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 54, 65547, v9);
  result = *(unsigned int *)(a3 + 20);
  if ( (result & 0x2000) == 0 )
  {
    if ( (result & 0x4000) == 0 )
      return result;
    return sub_3249FA0(a1, a2, 120);
  }
  sub_3249FA0(a1, a2, 119);
  result = *(unsigned int *)(a3 + 20);
  if ( (result & 0x4000) != 0 )
    return sub_3249FA0(a1, a2, 120);
  return result;
}
