// Function: sub_7E6B40
// Address: 0x7e6b40
//
__int64 __fastcall sub_7E6B40(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, _BOOL4 *a5)
{
  __int64 v5; // r13
  char v9; // al
  unsigned int v10; // r8d
  char v11; // r10
  __int64 v12; // r15
  __int64 v14; // r12
  _BOOL4 v15; // r8d
  _BOOL4 v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r15
  _BOOL4 v26; // eax
  char v27; // [rsp+Bh] [rbp-45h]
  _BOOL4 v28; // [rsp+Ch] [rbp-44h]
  _QWORD v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1;
  *a5 = 0;
  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
LABEL_12:
    v14 = *(_QWORD *)(v5 + 56);
    v15 = 1;
    if ( *(_BYTE *)(v14 + 173) == 6 && *(_BYTE *)(v14 + 176) == 2 )
      v15 = *(_BYTE *)(*(_QWORD *)(v14 + 184) + 173LL) != 2;
    v28 = v15;
    v16 = sub_70FCE0(*(_QWORD *)(v5 + 56));
    v10 = v28;
    if ( v16 )
    {
      v19 = sub_711520(v14, a2, v17, v18, v28);
      v10 = v28;
      v16 = v19 == 0;
    }
    *a5 = v16;
    return v10;
  }
  while ( 2 )
  {
    if ( sub_7E2090(v5) )
      return 1;
    v9 = *(_BYTE *)(v5 + 24);
    if ( v9 != 3 )
    {
      if ( v9 == 20 )
      {
        v26 = sub_70FCD0(*(_QWORD *)(v5 + 56));
        v10 = 1;
        *a5 = v26;
      }
      else
      {
        v10 = sub_731660(v5);
        if ( !v10 && *(_BYTE *)(v5 + 24) == 1 )
        {
          v11 = *(_BYTE *)(v5 + 56);
          v12 = *(_QWORD *)(v5 + 72);
          switch ( v11 )
          {
            case 0:
            case 6:
            case 7:
            case 8:
            case 9:
            case 21:
              goto LABEL_11;
            case 1:
            case 2:
            case 3:
            case 4:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
            case 17:
            case 18:
            case 19:
            case 20:
            case 22:
            case 23:
            case 24:
            case 25:
            case 26:
            case 27:
            case 28:
            case 29:
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
            case 41:
            case 42:
            case 43:
            case 44:
            case 45:
            case 46:
            case 47:
            case 48:
            case 49:
            case 52:
            case 53:
            case 54:
            case 55:
            case 56:
            case 57:
            case 58:
              return v10;
            case 5:
              if ( !(unsigned int)sub_8D2E30(*(_QWORD *)v5) )
                return 0;
              goto LABEL_11;
            case 50:
            case 51:
              v27 = *(_BYTE *)(v5 + 56);
              v10 = sub_7E6B40(*(_QWORD *)(v5 + 72), (unsigned int)a2, a3, a4, v29);
              if ( v10 )
              {
                v10 = sub_7E6B40(*(_QWORD *)(v12 + 16), (unsigned int)a2, a3, a4, (char *)v29 + 4);
                if ( v27 == 50 )
                  goto LABEL_33;
              }
              return v10;
            case 59:
              v20 = *(_QWORD *)(v12 + 16);
              if ( *(_BYTE *)(v20 + 24) != 2 )
                return 0;
              v21 = *(_QWORD *)(v20 + 56);
              if ( !sub_70FCE0(v21) || !(unsigned int)sub_711520(v21, a2, v22, v23, v24) )
                return 0;
LABEL_11:
              *a5 = 0;
              v5 = v12;
              if ( *(_BYTE *)(v12 + 24) == 2 )
                goto LABEL_12;
              continue;
            default:
              if ( v11 == 92 )
              {
                v10 = sub_7E6B40(*(_QWORD *)(v5 + 72), (unsigned int)a2, a3, a4, v29);
                if ( v10 )
                {
                  v10 = sub_7E6B40(*(_QWORD *)(v12 + 16), (unsigned int)a2, a3, a4, (char *)v29 + 4);
LABEL_33:
                  *a5 = v29[0] != 0;
                }
              }
              else if ( (unsigned __int8)(v11 - 94) <= 1u )
              {
                v10 = sub_7E6B40(*(_QWORD *)(v5 + 72), (unsigned int)a2, a3, a4, a5);
                if ( !*a5
                  && (*(_BYTE *)(v5 + 25) & 1) != 0
                  && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 56LL) + 128LL) )
                {
                  *a5 = 1;
                }
              }
              return v10;
          }
        }
        return 0;
      }
      return v10;
    }
    break;
  }
  v25 = *(_QWORD *)(v5 + 56);
  if ( (*(_BYTE *)(v5 + 25) & 1) != 0 )
  {
    *a5 = sub_70FCC0(*(_QWORD *)(v5 + 56));
    goto LABEL_27;
  }
  if ( (*(_DWORD *)(v25 + 168) & 0x10008000) == 0x8000 )
  {
LABEL_27:
    if ( (*(_BYTE *)(v25 + 172) & 1) != 0 )
    {
LABEL_28:
      if ( a4
        || qword_4F04C50
        && *(_QWORD *)(qword_4F04C50 + 64LL) == v25
        && (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 192LL) & 2) != 0 )
      {
        *a5 = 1;
      }
    }
    return 1;
  }
  if ( (*(_BYTE *)(v25 + 172) & 1) != 0 )
    goto LABEL_28;
  if ( (_DWORD)a2 )
    return a3 == 0;
  if ( (*(_BYTE *)(v25 + 89) & 1) == 0 )
    return a3 == 0;
  if ( (*(_BYTE *)(v25 + 169) & 0x40) != 0 )
    return a3 == 0;
  v10 = 1;
  if ( *(_BYTE *)(v25 + 136) <= 2u )
    return a3 == 0;
  return v10;
}
