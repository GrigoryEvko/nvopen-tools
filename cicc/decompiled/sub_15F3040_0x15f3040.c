// Function: sub_15F3040
// Address: 0x15f3040
//
__int64 __fastcall sub_15F3040(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rbp
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 i; // rax
  __int64 v14; // rdx
  unsigned __int16 v15; // ax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  _QWORD v25[6]; // [rsp-30h] [rbp-30h] BYREF

  v3 = *(unsigned __int8 *)(a1 + 16);
  v25[5] = v2;
  v25[2] = v1;
  switch ( v3 )
  {
    case 29:
      if ( (unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 36) )
        return 0;
      if ( *(char *)(a1 + 23) >= 0 )
        goto LABEL_45;
      v5 = sub_1648A40(a1);
      v7 = v5 + v6;
      v8 = 0;
      if ( *(char *)(a1 + 23) < 0 )
        v8 = sub_1648A40(a1);
      if ( !(unsigned int)((v7 - v8) >> 4) )
      {
LABEL_45:
        v9 = *(_QWORD *)(a1 - 72);
        if ( !*(_BYTE *)(v9 + 16) )
        {
          v25[0] = *(_QWORD *)(v9 + 112);
          if ( (unsigned __int8)sub_1560260(v25, -1, 36) )
            return 0;
        }
      }
      if ( (unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 37) )
        return 0;
      if ( *(char *)(a1 + 23) < 0 )
      {
        v10 = sub_1648A40(a1);
        v12 = v10 + v11;
        for ( i = *(char *)(a1 + 23) >= 0 ? 0LL : sub_1648A40(a1); v12 != i; i += 16 )
        {
          if ( *(_DWORD *)(*(_QWORD *)i + 8LL) > 1u )
            return 1;
        }
      }
      v14 = *(_QWORD *)(a1 - 72);
      result = 1;
      if ( !*(_BYTE *)(v14 + 16) )
        goto LABEL_39;
      return result;
    case 30:
    case 31:
    case 32:
    case 34:
    case 35:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 56:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 71:
    case 72:
    case 73:
    case 75:
    case 76:
    case 77:
    case 79:
    case 80:
    case 81:
      return 0;
    case 33:
    case 55:
    case 57:
    case 58:
    case 59:
    case 74:
    case 82:
      return 1;
    case 54:
      v15 = *(_WORD *)(a1 + 18);
      if ( ((v15 >> 7) & 6) != 0 )
        LOBYTE(v15) = 1;
      return v15 & 1;
    case 78:
      if ( (unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 36) )
        return 0;
      if ( *(char *)(a1 + 23) >= 0 )
        goto LABEL_46;
      v16 = sub_1648A40(a1);
      v18 = v16 + v17;
      v19 = 0;
      if ( *(char *)(a1 + 23) < 0 )
        v19 = sub_1648A40(a1);
      if ( !(unsigned int)((v18 - v19) >> 4) )
      {
LABEL_46:
        v20 = *(_QWORD *)(a1 - 24);
        if ( !*(_BYTE *)(v20 + 16) )
        {
          v25[0] = *(_QWORD *)(v20 + 112);
          if ( (unsigned __int8)sub_1560260(v25, -1, 36) )
            return 0;
        }
      }
      if ( (unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 37) )
        return 0;
      if ( *(char *)(a1 + 23) >= 0 )
        goto LABEL_38;
      v21 = sub_1648A40(a1);
      v23 = v21 + v22;
      v24 = *(char *)(a1 + 23) >= 0 ? 0LL : sub_1648A40(a1);
      if ( v24 == v23 )
        goto LABEL_38;
      break;
    default:
      return 0;
  }
  do
  {
    if ( *(_DWORD *)(*(_QWORD *)v24 + 8LL) > 1u )
      return 1;
    v24 += 16;
  }
  while ( v23 != v24 );
LABEL_38:
  v14 = *(_QWORD *)(a1 - 24);
  result = 1;
  if ( !*(_BYTE *)(v14 + 16) )
  {
LABEL_39:
    v25[0] = *(_QWORD *)(v14 + 112);
    return (unsigned int)sub_1560260(v25, -1, 37) ^ 1;
  }
  return result;
}
