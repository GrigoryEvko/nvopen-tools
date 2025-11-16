// Function: sub_14D90D0
// Address: 0x14d90d0
//
__int64 __fastcall sub_14D90D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  _DWORD *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  char v19; // al
  int v20; // eax
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  int v25; // eax
  char v26; // al
  unsigned __int64 v27; // rdx
  int v28; // eax
  int v29; // eax
  _QWORD v30[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2;
  v3 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (a1 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  if ( (a1 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 21)
      || (v10 = *(_QWORD *)(v3 - 24), !*(_BYTE *)(v10 + 16))
      && (v30[0] = *(_QWORD *)(v10 + 112), (unsigned __int8)sub_1560260(v30, 0xFFFFFFFFLL, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 5) )
      {
        v11 = *(_QWORD *)(v3 - 24);
        if ( *(_BYTE *)(v11 + 16) )
          return 0;
        v30[0] = *(_QWORD *)(v11 + 112);
        if ( !(unsigned __int8)sub_1560260(v30, 0xFFFFFFFFLL, 5) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 52) )
      return 0;
    v7 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v7 + 16) )
    {
LABEL_13:
      v8 = *(_DWORD *)(a2 + 36);
      switch ( v8 )
      {
        case 0u:
          if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
          {
            v15 = (_DWORD *)sub_1649960(a2);
            switch ( *(_BYTE *)v15 )
            {
              case '_':
                v19 = *((_BYTE *)v15 + 1);
                if ( v19 == 90 )
                {
                  if ( v14 <= 6 )
                    return 0;
                  v26 = *((_BYTE *)v15 + 2);
                  v27 = v14 - 3;
                  switch ( v26 )
                  {
                    case '3':
                      switch ( *((_BYTE *)v15 + 3) )
                      {
                        case 'c':
                          if ( v27 != 4 )
                            return 0;
                          v2 = 1;
                          if ( *(_DWORD *)((char *)v15 + 3) != 1718841187 )
                            LOBYTE(v2) = *(_DWORD *)((char *)v15 + 3) == 1685286755;
                          break;
                        case 'e':
                          if ( v27 != 4 )
                            return 0;
                          v2 = 1;
                          if ( *(_DWORD *)((char *)v15 + 3) != 1718646885 )
                            LOBYTE(v2) = *(_DWORD *)((char *)v15 + 3) == 1685092453;
                          break;
                        case 'l':
                          if ( v27 != 4 )
                            return 0;
                          v2 = 1;
                          if ( *(_DWORD *)((char *)v15 + 3) != 1718054764 )
                            LOBYTE(v2) = *(_DWORD *)((char *)v15 + 3) == 1684500332;
                          break;
                        case 'p':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "powff", 5u) )
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "powdd", 5u) == 0;
                          break;
                        case 's':
                          if ( v27 != 4 )
                            return 0;
                          v2 = 1;
                          if ( *(_DWORD *)((char *)v15 + 3) != 1718511987 )
                            LOBYTE(v2) = *(_DWORD *)((char *)v15 + 3) == 1684957555;
                          break;
                        case 't':
                          if ( v27 != 4 )
                            return 0;
                          v2 = 1;
                          if ( *(_DWORD *)((char *)v15 + 3) != 1718509940 )
                            LOBYTE(v2) = *(_DWORD *)((char *)v15 + 3) == 1684955508;
                          break;
                        default:
                          return 0;
                      }
                      break;
                    case '4':
                      switch ( *((_BYTE *)v15 + 3) )
                      {
                        case 'a':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "acosf", 5u)
                            && memcmp((char *)v15 + 3, "acosd", 5u)
                            && memcmp((char *)v15 + 3, "asinf", 5u)
                            && memcmp((char *)v15 + 3, "asind", 5u)
                            && memcmp((char *)v15 + 3, "atanf", 5u) )
                          {
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "atand", 5u) == 0;
                          }
                          return v2;
                        case 'c':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "ceilf", 5u)
                            && memcmp((char *)v15 + 3, "ceild", 5u)
                            && memcmp((char *)v15 + 3, "coshf", 5u) )
                          {
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "coshd", 5u) == 0;
                          }
                          return v2;
                        case 'e':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "exp2f", 5u) )
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "exp2d", 5u) == 0;
                          return v2;
                        case 'f':
                          if ( v27 == 5 )
                          {
                            v2 = 1;
                            if ( memcmp((char *)v15 + 3, "fabsf", 5u) )
                              LOBYTE(v2) = memcmp((char *)v15 + 3, "fabsd", 5u) == 0;
                          }
                          else
                          {
                            if ( v27 != 6 )
                              return 0;
                            v2 = 1;
                            if ( memcmp((char *)v15 + 3, "fmodff", 6u) )
                              LOBYTE(v2) = memcmp((char *)v15 + 3, "fmoddd", 6u) == 0;
                          }
                          break;
                        case 's':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "sinhf", 5u)
                            && memcmp((char *)v15 + 3, "sinhd", 5u)
                            && memcmp((char *)v15 + 3, "sqrtf", 5u) )
                          {
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "sqrtd", 5u) == 0;
                          }
                          return v2;
                        case 't':
                          if ( v27 != 5 )
                            return 0;
                          v2 = 1;
                          if ( memcmp((char *)v15 + 3, "tanhf", 5u) )
                            LOBYTE(v2) = memcmp((char *)v15 + 3, "tanhd", 5u) == 0;
                          return v2;
                        default:
                          return 0;
                      }
                      break;
                    case '5':
                      if ( v27 == 7 )
                      {
                        v2 = 1;
                        if ( memcmp((char *)v15 + 3, "atan2ff", 7u) )
                          LOBYTE(v2) = memcmp((char *)v15 + 3, "atan2dd", 7u) == 0;
                      }
                      else
                      {
                        if ( v27 != 6 )
                          return 0;
                        v2 = 1;
                        if ( memcmp((char *)v15 + 3, "floorf", 6u)
                          && memcmp((char *)v15 + 3, "floord", 6u)
                          && memcmp((char *)v15 + 3, "log10f", 6u) )
                        {
                          LOBYTE(v2) = memcmp((char *)v15 + 3, "log10d", 6u) == 0;
                        }
                      }
                      break;
                    default:
                      return 0;
                  }
                }
                else
                {
                  if ( v14 <= 0xB || v19 != 95 )
                    return 0;
                  switch ( *((_BYTE *)v15 + 2) )
                  {
                    case 'a':
                      switch ( v14 )
                      {
                        case 0xDuLL:
                          v2 = 1;
                          if ( memcmp(v15, "__acos_finite", 0xDu) )
                            LOBYTE(v2) = memcmp(v15, "__asin_finite", 0xDu) == 0;
                          break;
                        case 0xEuLL:
                          v2 = 1;
                          if ( memcmp(v15, "__acosf_finite", 0xEu) )
                          {
                            if ( memcmp(v15, "__asinf_finite", 0xEu) )
                              LOBYTE(v2) = memcmp(v15, "__atan2_finite", 0xEu) == 0;
                          }
                          break;
                        case 0xFuLL:
                          LOBYTE(v2) = memcmp(v15, "__atan2f_finite", 0xFu) == 0;
                          break;
                        default:
                          return 0;
                      }
                      return v2;
                    case 'c':
                      if ( v14 == 13 )
                      {
                        LOBYTE(v2) = memcmp(v15, "__cosh_finite", 0xDu) == 0;
                      }
                      else
                      {
                        if ( v14 != 14 )
                          return 0;
                        LOBYTE(v2) = memcmp(v15, "__coshf_finite", 0xEu) == 0;
                      }
                      return v2;
                    case 'e':
                      switch ( v14 )
                      {
                        case 0xCuLL:
                          LOBYTE(v2) = memcmp(v15, "__exp_finite", 0xCu) == 0;
                          break;
                        case 0xDuLL:
                          v2 = 1;
                          if ( memcmp(v15, "__expf_finite", 0xDu) )
                            LOBYTE(v2) = memcmp(v15, "__exp2_finite", 0xDu) == 0;
                          break;
                        case 0xEuLL:
                          LOBYTE(v2) = memcmp(v15, "__exp2f_finite", 0xEu) == 0;
                          break;
                        default:
                          return 0;
                      }
                      return v2;
                    case 'l':
                      switch ( v14 )
                      {
                        case 0xCuLL:
                          LOBYTE(v2) = memcmp(v15, "__log_finite", 0xCu) == 0;
                          return v2;
                        case 0xDuLL:
                          LOBYTE(v2) = memcmp(v15, "__logf_finite", 0xDu) == 0;
                          return v2;
                        case 0xEuLL:
                          LOBYTE(v2) = memcmp(v15, "__log10_finite", 0xEu) == 0;
                          return v2;
                      }
                      if ( v14 != 15 )
                        return 0;
                      LOBYTE(v2) = memcmp(v15, "__log10f_finite", 0xFu) == 0;
                      return v2;
                    case 'p':
                      if ( v14 == 12 )
                      {
                        LOBYTE(v2) = memcmp(v15, "__pow_finite", 0xCu) == 0;
                      }
                      else
                      {
                        if ( v14 != 13 )
                          return 0;
                        LOBYTE(v2) = memcmp(v15, "__powf_finite", 0xDu) == 0;
                      }
                      return v2;
                    case 's':
                      if ( v14 == 13 )
                      {
                        LOBYTE(v2) = memcmp(v15, "__sinh_finite", 0xDu) == 0;
                      }
                      else
                      {
                        if ( v14 != 14 )
                          return 0;
                        LOBYTE(v2) = memcmp(v15, "__sinhf_finite", 0xEu) == 0;
                      }
                      break;
                    default:
                      return 0;
                  }
                }
                return v2;
              case 'a':
                switch ( v14 )
                {
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 1936679777 && *v15 != 1852404577 )
                      LOBYTE(v2) = *v15 == 1851880545;
                    break;
                  case 5uLL:
                    v2 = 1;
                    if ( memcmp(v15, "atan2", 5u) && memcmp(v15, "acosf", 5u) && memcmp(v15, "asinf", 5u) )
                      LOBYTE(v2) = memcmp(v15, "atanf", 5u) == 0;
                    break;
                  case 6uLL:
                    LOBYTE(v2) = memcmp(v15, "atan2f", 6u) == 0;
                    break;
                  default:
                    return 0;
                }
                return v2;
              case 'c':
                switch ( v14 )
                {
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 1818846563 && *v15 != 1752395619 )
                      LOBYTE(v2) = *v15 == 1718841187;
                    break;
                  case 3uLL:
                    if ( *(_WORD *)v15 != 28515 || (v28 = 0, *((_BYTE *)v15 + 2) != 115) )
                      v28 = 1;
                    LOBYTE(v2) = v28 == 0;
                    break;
                  case 5uLL:
                    v2 = 1;
                    if ( memcmp(v15, "ceilf", 5u) )
                      LOBYTE(v2) = memcmp(v15, "coshf", 5u) == 0;
                    break;
                  default:
                    return 0;
                }
                return v2;
              case 'e':
                switch ( v14 )
                {
                  case 3uLL:
                    if ( *(_WORD *)v15 != 30821 || (v22 = 0, *((_BYTE *)v15 + 2) != 112) )
                      v22 = 1;
                    LOBYTE(v2) = v22 == 0;
                    break;
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 846231653 )
                      LOBYTE(v2) = *v15 == 1718646885;
                    break;
                  case 5uLL:
                    LOBYTE(v2) = memcmp(v15, "exp2f", 5u) == 0;
                    break;
                  default:
                    return 0;
                }
                return v2;
              case 'f':
                switch ( v14 )
                {
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 1935827302 )
                      LOBYTE(v2) = *v15 == 1685024102;
                    break;
                  case 5uLL:
                    if ( *v15 != 1869573222 || (v2 = 1, *((_BYTE *)v15 + 4) != 114) )
                    {
                      v2 = 1;
                      if ( memcmp(v15, "fabsf", 5u) )
                        LOBYTE(v2) = memcmp(v15, "fmodf", 5u) == 0;
                    }
                    break;
                  case 6uLL:
                    LOBYTE(v2) = memcmp(v15, "floorf", 6u) == 0;
                    break;
                  default:
                    return 0;
                }
                return v2;
              case 'l':
                switch ( v14 )
                {
                  case 3uLL:
                    if ( *(_WORD *)v15 != 28524 || (v24 = 0, *((_BYTE *)v15 + 2) != 103) )
                      v24 = 1;
                    LOBYTE(v2) = v24 == 0;
                    return v2;
                  case 5uLL:
                    if ( *v15 != 828862316 || (v29 = 0, *((_BYTE *)v15 + 4) != 48) )
                      v29 = 1;
                    LOBYTE(v2) = v29 == 0;
                    return v2;
                  case 4uLL:
                    LOBYTE(v2) = *v15 == 1718054764;
                    return v2;
                  case 6uLL:
                    LOBYTE(v2) = memcmp(v15, "log10f", 6u) == 0;
                    return v2;
                }
                return 0;
              case 'p':
                if ( v14 == 3 )
                {
                  if ( *(_WORD *)v15 != 28528 || (v20 = 0, *((_BYTE *)v15 + 2) != 119) )
                    v20 = 1;
                  LOBYTE(v2) = v20 == 0;
                }
                else
                {
                  if ( v14 != 4 )
                    return 0;
                  LOBYTE(v2) = *v15 == 1719103344;
                }
                return v2;
              case 'r':
                if ( v14 == 5 )
                {
                  if ( *v15 != 1853190002 || (v25 = 0, *((_BYTE *)v15 + 4) != 100) )
                    v25 = 1;
                  LOBYTE(v2) = v25 == 0;
                }
                else
                {
                  if ( v14 != 6 )
                    return 0;
                  if ( *v15 != 1853190002 || (v18 = 0, *((_WORD *)v15 + 2) != 26212) )
                    v18 = 1;
                  LOBYTE(v2) = v18 == 0;
                }
                return v2;
              case 's':
                switch ( v14 )
                {
                  case 3uLL:
                    if ( *(_WORD *)v15 != 26995 || (v21 = 0, *((_BYTE *)v15 + 2) != 110) )
                      v21 = 1;
                    LOBYTE(v2) = v21 == 0;
                    break;
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 1752066419 && *v15 != 1953657203 )
                      LOBYTE(v2) = *v15 == 1718511987;
                    break;
                  case 5uLL:
                    v2 = 1;
                    if ( memcmp(v15, "sinhf", 5u) )
                      LOBYTE(v2) = memcmp(v15, "sqrtf", 5u) == 0;
                    break;
                  default:
                    return 0;
                }
                return v2;
              case 't':
                switch ( v14 )
                {
                  case 3uLL:
                    if ( *(_WORD *)v15 != 24948 || (v23 = 0, *((_BYTE *)v15 + 2) != 110) )
                      v23 = 1;
                    LOBYTE(v2) = v23 == 0;
                    break;
                  case 4uLL:
                    v2 = 1;
                    if ( *v15 != 1752064372 )
                      LOBYTE(v2) = *v15 == 1718509940;
                    break;
                  case 5uLL:
                    LOBYTE(v2) = memcmp(v15, "tanhf", 5u) == 0;
                    break;
                  default:
                    return 0;
                }
                break;
              default:
                return 0;
            }
          }
          else
          {
            return 0;
          }
          return v2;
        case 1u:
        case 2u:
        case 3u:
        case 4u:
        case 7u:
        case 9u:
        case 0xAu:
        case 0xEu:
        case 0xFu:
        case 0x10u:
        case 0x11u:
        case 0x12u:
        case 0x13u:
        case 0x14u:
        case 0x15u:
        case 0x16u:
        case 0x17u:
        case 0x18u:
        case 0x19u:
        case 0x1Au:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Du:
        case 0x22u:
        case 0x23u:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x28u:
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Cu:
        case 0x2Du:
        case 0x2Eu:
        case 0x2Fu:
        case 0x30u:
        case 0x31u:
        case 0x32u:
        case 0x33u:
        case 0x34u:
        case 0x35u:
        case 0x38u:
        case 0x39u:
        case 0x3Au:
        case 0x3Bu:
        case 0x3Cu:
        case 0x3Du:
        case 0x3Eu:
        case 0x3Fu:
        case 0x40u:
        case 0x41u:
        case 0x42u:
        case 0x43u:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
        case 0x48u:
        case 0x49u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x55u:
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x59u:
        case 0x5Au:
        case 0x5Bu:
        case 0x5Cu:
        case 0x5Du:
        case 0x5Eu:
        case 0x5Fu:
        case 0x62u:
        case 0x65u:
        case 0x66u:
        case 0x67u:
        case 0x68u:
        case 0x69u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
        case 0x6Eu:
        case 0x6Fu:
        case 0x70u:
        case 0x71u:
        case 0x72u:
        case 0x74u:
        case 0x75u:
        case 0x76u:
        case 0x77u:
        case 0x78u:
        case 0x79u:
        case 0x7Du:
        case 0x7Eu:
        case 0x7Fu:
        case 0x80u:
        case 0x82u:
        case 0x83u:
        case 0x85u:
        case 0x86u:
        case 0x87u:
        case 0x88u:
        case 0x89u:
        case 0x8Au:
        case 0x8Du:
        case 0x8Eu:
        case 0x8Fu:
        case 0x90u:
        case 0x91u:
        case 0x94u:
        case 0x95u:
        case 0x96u:
        case 0x97u:
        case 0x98u:
        case 0x99u:
        case 0x9Au:
        case 0x9Bu:
        case 0x9Cu:
        case 0x9Du:
        case 0x9Eu:
        case 0x9Fu:
        case 0xA0u:
        case 0xA1u:
        case 0xA2u:
        case 0xA3u:
        case 0xA4u:
        case 0xA5u:
        case 0xA6u:
        case 0xA7u:
        case 0xA8u:
        case 0xA9u:
        case 0xAAu:
        case 0xABu:
        case 0xACu:
        case 0xADu:
        case 0xAEu:
        case 0xAFu:
        case 0xB0u:
        case 0xB1u:
        case 0xB2u:
        case 0xB3u:
        case 0xB4u:
        case 0xB5u:
        case 0xB6u:
        case 0xB7u:
        case 0xB8u:
        case 0xB9u:
        case 0xBAu:
        case 0xBEu:
        case 0xBFu:
        case 0xC0u:
        case 0xC1u:
        case 0xC5u:
        case 0xC7u:
        case 0xC8u:
        case 0xC9u:
        case 0xCAu:
        case 0xCCu:
        case 0xCDu:
        case 0xCFu:
        case 0xD0u:
          return 0;
        case 5u:
        case 6u:
        case 8u:
        case 0xBu:
        case 0xCu:
        case 0xDu:
        case 0x1Eu:
        case 0x1Fu:
        case 0x20u:
        case 0x21u:
        case 0x36u:
        case 0x37u:
        case 0x60u:
        case 0x61u:
        case 0x63u:
        case 0x64u:
        case 0x73u:
        case 0x7Au:
        case 0x7Bu:
        case 0x7Cu:
        case 0x81u:
        case 0x84u:
        case 0x8Bu:
        case 0x8Cu:
        case 0x92u:
        case 0x93u:
        case 0xBBu:
        case 0xBCu:
        case 0xBDu:
        case 0xC2u:
        case 0xC3u:
        case 0xC4u:
        case 0xC6u:
        case 0xCBu:
        case 0xCEu:
        case 0xD1u:
        case 0xD2u:
        case 0xD3u:
          return 1;
        default:
          if ( v8 > 0xFEA )
          {
            if ( v8 > 0x1184 )
            {
              if ( v8 <= 0x1CAC )
              {
                if ( v8 > 0x1C80 )
                {
                  v17 = 0xC6000000033LL;
                  if ( _bittest64(&v17, v8 - 7297) )
                    return 1;
                }
                else
                {
                  v2 = 1;
                  if ( v8 == 5293 || v8 == 5300 )
                    return v2;
                }
              }
            }
            else if ( v8 > 0x117A )
            {
              v2 = 1;
              if ( ((1LL << ((unsigned __int8)v8 - 123)) & 0x309) != 0 )
                return v2;
            }
            else if ( v8 > 0x104C )
            {
              v2 = 1;
              if ( v8 > 0x1087 )
              {
                if ( v8 == 4413 )
                  return v2;
              }
              else if ( v8 > 0x1085 )
              {
                return v2;
              }
            }
            else
            {
              if ( v8 > 0x104A )
                return 1;
              v12 = v8 - 4114;
              if ( (unsigned int)v12 <= 0x36 )
              {
                v13 = 0x60000000000033LL;
                if ( _bittest64(&v13, v12) )
                  return 1;
              }
            }
          }
          else
          {
            if ( v8 > 0xFE8 )
              return 1;
            if ( v8 <= 0xF72 )
            {
              if ( v8 > 0xF1D )
              {
                switch ( v8 )
                {
                  case 0xF1Eu:
                  case 0xF1Fu:
                  case 0xF47u:
                  case 0xF48u:
                  case 0xF54u:
                  case 0xF55u:
                  case 0xF68u:
                  case 0xF6Au:
                  case 0xF6Cu:
                  case 0xF6Eu:
                  case 0xF70u:
                  case 0xF72u:
                    return 1;
                  default:
                    return 0;
                }
              }
              if ( v8 <= 0xEE3 )
              {
                if ( v8 > 0xEB3 )
                {
                  v16 = 0x80000080C003LL;
                  if ( _bittest64(&v16, v8 - 3764) )
                    return 1;
                }
                else
                {
                  v2 = 1;
                  if ( v8 > 0xE37 )
                  {
                    if ( v8 == 3660 )
                      return v2;
                  }
                  else if ( v8 > 0xE34 )
                  {
                    return v2;
                  }
                }
              }
            }
          }
          return 0;
      }
    }
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 21)
      || (v6 = *(_QWORD *)(v3 - 72), !*(_BYTE *)(v6 + 16))
      && (v30[0] = *(_QWORD *)(v6 + 112), (unsigned __int8)sub_1560260(v30, 0xFFFFFFFFLL, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 5) )
      {
        v9 = *(_QWORD *)(v3 - 72);
        if ( *(_BYTE *)(v9 + 16) )
          return 0;
        v30[0] = *(_QWORD *)(v9 + 112);
        if ( !(unsigned __int8)sub_1560260(v30, 0xFFFFFFFFLL, 5) )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560260(v4, 0xFFFFFFFFLL, 52) )
      return 0;
    v7 = *(_QWORD *)(v3 - 72);
    if ( *(_BYTE *)(v7 + 16) )
      goto LABEL_13;
  }
  v30[0] = *(_QWORD *)(v7 + 112);
  if ( !(unsigned __int8)sub_1560260(v30, 0xFFFFFFFFLL, 52) )
    goto LABEL_13;
  return 0;
}
