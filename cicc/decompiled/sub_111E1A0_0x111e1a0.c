// Function: sub_111E1A0
// Address: 0x111e1a0
//
__int64 __fastcall sub_111E1A0(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // r15
  __int64 i; // r12
  unsigned __int8 *v5; // r13
  int v6; // ecx
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx
  _BYTE *v14; // rdi
  _QWORD *v15[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v16; // [rsp+30h] [rbp-40h]

  v2 = *(_WORD *)(a2 + 2) & 0x3F;
  switch ( v2 )
  {
    case 3u:
    case 5u:
    case 6u:
    case 0x21u:
    case 0x23u:
    case 0x25u:
    case 0x27u:
    case 0x29u:
      for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v5 = *(unsigned __int8 **)(i + 24);
        if ( v5 )
        {
          v6 = *v5;
          switch ( v6 )
          {
            case 59:
              v15[0] = 0;
              if ( *v5 != 59
                || !(unsigned __int8)sub_995B10(v15, *((_QWORD *)v5 - 8))
                && !(unsigned __int8)sub_995B10(v15, *((_QWORD *)v5 - 4)) )
              {
                return 0;
              }
              break;
            case 86:
              if ( (unsigned int)sub_BD2910(i) )
                return 0;
              v9 = *((_QWORD *)v5 + 1);
              if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
                v9 = **(_QWORD **)(v9 + 16);
              if ( !sub_BCAC40(v9, 1) )
                goto LABEL_22;
              if ( *v5 == 57 )
                return 0;
              v10 = *((_QWORD *)v5 + 1);
              if ( *v5 == 86 && *(_QWORD *)(*((_QWORD *)v5 - 12) + 8LL) == v10 && **((_BYTE **)v5 - 4) <= 0x15u )
              {
                if ( sub_AC30F0(*((_QWORD *)v5 - 4)) )
                  return 0;
LABEL_22:
                v10 = *((_QWORD *)v5 + 1);
              }
              if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
                v10 = **(_QWORD **)(v10 + 16);
              if ( sub_BCAC40(v10, 1) )
              {
                if ( *v5 == 58 )
                  return 0;
                if ( *v5 == 86 )
                {
                  v13 = *((_QWORD *)v5 + 1);
                  if ( *(_QWORD *)(*((_QWORD *)v5 - 12) + 8LL) == v13 )
                  {
                    v14 = (_BYTE *)*((_QWORD *)v5 - 8);
                    if ( *v14 <= 0x15u && sub_AD7A80(v14, 1, v13, v11, v12) )
                      return 0;
                  }
                }
              }
              continue;
            case 31:
              break;
            default:
              return 0;
          }
        }
      }
      *(_WORD *)(a2 + 2) = sub_B52870(v2) | *(_WORD *)(a2 + 2) & 0xFFC0;
      v15[0] = sub_BD5D20(a2);
      v15[2] = ".not";
      v16 = 773;
      v15[1] = v8;
      sub_BD6B50((unsigned __int8 *)a2, (const char **)v15);
      sub_F16650(a1, a2, 0);
      return a2;
    default:
      return 0;
  }
}
