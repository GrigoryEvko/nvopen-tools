// Function: sub_971E80
// Address: 0x971e80
//
char __fastcall sub_971E80(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r13d
  unsigned int v5; // ecx
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // r15
  size_t v12; // r14
  char *v13; // rdx
  __int64 v14; // rcx
  const void *v15; // rdi
  __int64 v16; // rax
  _BYTE *v17; // r9
  size_t v18; // r8
  _QWORD *v19; // r12
  _QWORD *v20; // rdi
  __int64 v21; // rax
  char v22; // al
  bool v23; // al
  bool v24; // al
  size_t v25; // [rsp+0h] [rbp-90h]
  _BYTE *v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD dest[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30; // [rsp+40h] [rbp-50h]
  __int64 v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h]

  v2 = a1 + 72;
  if ( (unsigned __int8)sub_A73ED0(a1 + 72, 23) || (unsigned __int8)sub_B49560(a1, 23) )
  {
    if ( ((unsigned __int8)sub_A73ED0(v2, 4) || (unsigned __int8)sub_B49560(a1, 4))
      && *(_QWORD *)(a1 + 80) == *(_QWORD *)(a2 + 24) )
    {
      goto LABEL_6;
    }
  }
  else if ( *(_QWORD *)(a1 + 80) == *(_QWORD *)(a2 + 24) )
  {
LABEL_6:
    v5 = *(_DWORD *)(a2 + 36);
    switch ( v5 )
    {
      case 0u:
        if ( (*(_BYTE *)(a2 + 7) & 0x10) == 0
          || (unsigned __int8)sub_A73ED0(v2, 72)
          || (unsigned __int8)sub_B49560(a1, 72) )
        {
          break;
        }
        v11 = (_BYTE *)sub_BD5D20(a2);
        v12 = v10;
        switch ( *v11 )
        {
          case '_':
            v16 = sub_B43CA0(a1);
            v28[0] = dest;
            v18 = *(_QWORD *)(v16 + 240);
            v19 = (_QWORD *)v16;
            v17 = *(_BYTE **)(v16 + 232);
            LOBYTE(v3) = v17 == 0 && &v17[v18] != 0;
            if ( (_BYTE)v3 )
              sub_426248((__int64)"basic_string::_M_construct null not valid");
            v27 = *(_QWORD *)(v16 + 240);
            if ( v18 > 0xF )
            {
              v25 = v18;
              v26 = v17;
              v21 = sub_22409D0(v28, &v27, 0);
              v17 = v26;
              v18 = v25;
              v28[0] = v21;
              v20 = (_QWORD *)v21;
              dest[0] = v27;
              goto LABEL_167;
            }
            if ( v18 == 1 )
            {
              LOBYTE(dest[0]) = *v17;
              goto LABEL_158;
            }
            if ( v18 )
            {
              v20 = dest;
LABEL_167:
              memcpy(v20, v17, v18);
            }
LABEL_158:
            v28[1] = v27;
            *(_BYTE *)(v28[0] + v27) = 0;
            v30 = v19[33];
            v31 = v19[34];
            v32 = v19[35];
            if ( (unsigned int)(v30 - 42) > 1 || v11[1] != 90 || v12 <= 6 )
            {
              if ( (_QWORD *)v28[0] != dest )
                j_j___libc_free_0(v28[0], dest[0] + 1LL);
              if ( v12 > 0xB && v11[1] == 95 )
              {
                switch ( v11[2] )
                {
                  case 'a':
                    if ( sub_9691B0(v11, v12, "__acos_finite", 13)
                      || sub_9691B0(v11, v12, "__acosf_finite", 14)
                      || sub_9691B0(v11, v12, "__asin_finite", 13)
                      || sub_9691B0(v11, v12, "__asinf_finite", 14)
                      || sub_9691B0(v11, v12, "__atan2_finite", 14) )
                    {
                      goto LABEL_27;
                    }
                    v13 = "__atan2f_finite";
                    v14 = 15;
                    break;
                  case 'c':
                    v24 = sub_9691B0(v11, v12, "__cosh_finite", 13);
                    v13 = "__coshf_finite";
                    LOBYTE(v3) = v24;
                    if ( v24 )
                      return v3;
                    goto LABEL_213;
                  case 'e':
                    if ( sub_9691B0(v11, v12, "__exp_finite", 12) )
                      goto LABEL_27;
                    if ( sub_9691B0(v11, v12, "__expf_finite", 13) )
                      goto LABEL_27;
                    v23 = sub_9691B0(v11, v12, "__exp2_finite", 13);
                    v13 = "__exp2f_finite";
                    if ( v23 )
                      goto LABEL_27;
                    goto LABEL_213;
                  case 'l':
                    if ( sub_9691B0(v11, v12, "__log_finite", 12)
                      || sub_9691B0(v11, v12, "__logf_finite", 13)
                      || sub_9691B0(v11, v12, "__log10_finite", 14) )
                    {
                      goto LABEL_27;
                    }
                    v13 = "__log10f_finite";
                    v14 = 15;
                    break;
                  case 'p':
                    LOBYTE(v3) = sub_9691B0(v11, v12, "__pow_finite", 12);
                    if ( (_BYTE)v3 )
                      return v3;
                    v13 = "__powf_finite";
                    v14 = 13;
                    break;
                  case 's':
                    LOBYTE(v3) = sub_9691B0(v11, v12, "__sinh_finite", 13);
                    if ( (_BYTE)v3 )
                      return v3;
                    v13 = "__sinhf_finite";
LABEL_213:
                    v14 = 14;
                    break;
                  default:
                    goto LABEL_4;
                }
LABEL_87:
                v15 = v11;
                return sub_9691B0(v15, v12, v13, v14);
              }
              goto LABEL_4;
            }
            if ( (_QWORD *)v28[0] != dest )
              j_j___libc_free_0(v28[0], dest[0] + 1LL);
            v22 = v11[2];
            v12 -= 3LL;
            if ( v22 == 51 )
            {
              switch ( v11[3] )
              {
                case 'c':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "cosf", 4);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "cosd";
                  v14 = 4;
                  v15 = v11 + 3;
                  break;
                case 'e':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "expf", 4);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "expd";
                  v14 = 4;
                  v15 = v11 + 3;
                  break;
                case 'l':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "logf", 4);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "logd";
                  v14 = 4;
                  v15 = v11 + 3;
                  break;
                case 'p':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "powff", 5);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "powdd";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                case 's':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "sinf", 4);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "sind";
                  v14 = 4;
                  v15 = v11 + 3;
                  break;
                case 't':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "tanf", 4);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "tand";
                  v14 = 4;
                  v15 = v11 + 3;
                  break;
                default:
                  return v3;
              }
              return sub_9691B0(v15, v12, v13, v14);
            }
            if ( v22 == 52 )
            {
              switch ( v11[3] )
              {
                case 'a':
                  if ( sub_9691B0(v11 + 3, v12, "acosf", 5)
                    || sub_9691B0(v11 + 3, v12, "acosd", 5)
                    || sub_9691B0(v11 + 3, v12, "asinf", 5)
                    || sub_9691B0(v11 + 3, v12, "asind", 5)
                    || sub_9691B0(v11 + 3, v12, "atanf", 5) )
                  {
                    goto LABEL_27;
                  }
                  v13 = "atand";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                case 'c':
                  if ( sub_9691B0(v11 + 3, v12, "ceilf", 5)
                    || sub_9691B0(v11 + 3, v12, "ceild", 5)
                    || sub_9691B0(v11 + 3, v12, "coshf", 5) )
                  {
                    goto LABEL_27;
                  }
                  v13 = "coshd";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                case 'e':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "exp2f", 5);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "exp2d";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                case 'f':
                  if ( sub_9691B0(v11 + 3, v12, "fabsf", 5)
                    || sub_9691B0(v11 + 3, v12, "fabsd", 5)
                    || sub_9691B0(v11 + 3, v12, "fmodff", 6) )
                  {
                    goto LABEL_27;
                  }
                  v13 = "fmoddd";
                  v14 = 6;
                  v15 = v11 + 3;
                  break;
                case 's':
                  if ( sub_9691B0(v11 + 3, v12, "sinhf", 5)
                    || sub_9691B0(v11 + 3, v12, "sinhd", 5)
                    || sub_9691B0(v11 + 3, v12, "sqrtf", 5) )
                  {
                    goto LABEL_27;
                  }
                  v13 = "sqrtd";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                case 't':
                  LOBYTE(v3) = sub_9691B0(v11 + 3, v12, "tanhf", 5);
                  if ( (_BYTE)v3 )
                    return v3;
                  v13 = "tanhd";
                  v14 = 5;
                  v15 = v11 + 3;
                  break;
                default:
                  return v3;
              }
              return sub_9691B0(v15, v12, v13, v14);
            }
            if ( v22 != 53 )
              return v3;
            if ( !sub_9691B0(v11 + 3, v12, "atan2ff", 7)
              && !sub_9691B0(v11 + 3, v12, "atan2dd", 7)
              && !sub_9691B0(v11 + 3, v12, "floorf", 6)
              && !sub_9691B0(v11 + 3, v12, "floord", 6)
              && !sub_9691B0(v11 + 3, v12, "log10f", 6) )
            {
              v13 = "log10d";
              v14 = 6;
              v15 = v11 + 3;
              return sub_9691B0(v15, v12, v13, v14);
            }
            break;
          case 'a':
            if ( v10 == 4 && *(_DWORD *)v11 == 1936679777
              || sub_9691B0(v11, v10, "acosf", 5)
              || sub_9691B0(v11, v12, "asin", 4)
              || sub_9691B0(v11, v12, "asinf", 5)
              || sub_9691B0(v11, v12, "atan", 4)
              || sub_9691B0(v11, v12, "atanf", 5)
              || sub_9691B0(v11, v12, "atan2", 5) )
            {
              goto LABEL_27;
            }
            v13 = "atan2f";
            v14 = 6;
            goto LABEL_87;
          case 'c':
            if ( v10 == 4 && *(_DWORD *)v11 == 1818846563
              || sub_9691B0(v11, v10, "ceilf", 5)
              || sub_9691B0(v11, v12, "cos", 3)
              || sub_9691B0(v11, v12, "cosf", 4)
              || sub_9691B0(v11, v12, "cosh", 4) )
            {
              goto LABEL_27;
            }
            v13 = "coshf";
            v14 = 5;
            goto LABEL_87;
          case 'e':
            if ( v10 == 3 && !memcmp(v11, "exp", 3u)
              || sub_9691B0(v11, v12, "expf", 4)
              || sub_9691B0(v11, v12, "exp2", 4)
              || sub_9691B0(v11, v12, "exp2f", 5)
              || sub_9691B0(v11, v12, "erf", 3) )
            {
              goto LABEL_27;
            }
            v13 = "erff";
            v14 = 4;
            goto LABEL_87;
          case 'f':
            if ( v10 == 4 && *(_DWORD *)v11 == 1935827302
              || sub_9691B0(v11, v10, "fabsf", 5)
              || sub_9691B0(v11, v12, "floor", 5)
              || sub_9691B0(v11, v12, "floorf", 6)
              || sub_9691B0(v11, v12, "fmod", 4) )
            {
              goto LABEL_27;
            }
            v13 = "fmodf";
            v14 = 5;
            goto LABEL_87;
          case 'i':
            if ( v10 == 5 )
            {
              LOBYTE(v3) = 1;
              if ( !memcmp(v11, "ilogb", 5u) )
                return v3;
            }
            v13 = "ilogbf";
            v14 = 6;
            goto LABEL_87;
          case 'l':
            if ( v10 == 3 && !memcmp(v11, "log", 3u)
              || sub_9691B0(v11, v12, "logf", 4)
              || sub_9691B0(v11, v12, "logl", 4)
              || sub_9691B0(v11, v12, "log2", 4)
              || sub_9691B0(v11, v12, "log2f", 5)
              || sub_9691B0(v11, v12, "log10", 5)
              || sub_9691B0(v11, v12, "log10f", 6)
              || sub_9691B0(v11, v12, "logb", 4)
              || sub_9691B0(v11, v12, "logbf", 5)
              || sub_9691B0(v11, v12, "log1p", 5) )
            {
              goto LABEL_27;
            }
            v13 = "log1pf";
            v14 = 6;
            goto LABEL_87;
          case 'n':
            if ( v10 == 9 )
            {
              LOBYTE(v3) = 1;
              if ( !memcmp(v11, "nearbyint", 9u) )
                return v3;
            }
            v13 = "nearbyintf";
            v14 = 10;
            goto LABEL_87;
          case 'p':
            if ( v10 == 3 )
            {
              LOBYTE(v3) = 1;
              if ( !memcmp(v11, "pow", 3u) )
                return v3;
            }
            v13 = "powf";
            v14 = 4;
            goto LABEL_87;
          case 'r':
            if ( v10 == 9 && !memcmp(v11, "remainder", 9u)
              || sub_9691B0(v11, v12, "remainderf", 10)
              || sub_9691B0(v11, v12, "rint", 4)
              || sub_9691B0(v11, v12, "rintf", 5)
              || sub_9691B0(v11, v12, "round", 5) )
            {
              goto LABEL_27;
            }
            v13 = "roundf";
            v14 = 6;
            goto LABEL_87;
          case 's':
            if ( v10 == 3 && !memcmp(v11, "sin", 3u)
              || sub_9691B0(v11, v12, "sinf", 4)
              || sub_9691B0(v11, v12, "sinh", 4)
              || sub_9691B0(v11, v12, "sinhf", 5)
              || sub_9691B0(v11, v12, "sqrt", 4) )
            {
              goto LABEL_27;
            }
            v13 = "sqrtf";
            v14 = 5;
            goto LABEL_87;
          case 't':
            if ( v10 == 3 && !memcmp(v11, "tan", 3u)
              || sub_9691B0(v11, v12, "tanf", 4)
              || sub_9691B0(v11, v12, "tanh", 4)
              || sub_9691B0(v11, v12, "tanhf", 5)
              || sub_9691B0(v11, v12, "trunc", 5) )
            {
              goto LABEL_27;
            }
            v13 = "truncf";
            v14 = 6;
            goto LABEL_87;
          default:
            goto LABEL_4;
        }
        goto LABEL_27;
      case 1u:
      case 0xEu:
      case 0xFu:
      case 0x14u:
      case 0x15u:
      case 0x1Au:
      case 0x41u:
      case 0x42u:
      case 0x43u:
      case 0x61u:
      case 0x66u:
      case 0x67u:
      case 0x68u:
      case 0x69u:
      case 0x6Au:
      case 0x6Bu:
      case 0x6Cu:
      case 0x6Du:
      case 0x72u:
      case 0x73u:
      case 0x80u:
      case 0x83u:
      case 0x84u:
      case 0x85u:
      case 0x8Cu:
      case 0xAAu:
      case 0xACu:
      case 0xB4u:
      case 0xB5u:
      case 0xB9u:
      case 0xCEu:
      case 0xCFu:
      case 0xD0u:
      case 0xE4u:
      case 0xFAu:
      case 0x134u:
      case 0x135u:
      case 0x136u:
      case 0x137u:
      case 0x138u:
      case 0x139u:
      case 0x149u:
      case 0x14Au:
      case 0x14Bu:
      case 0x14Cu:
      case 0x14Du:
      case 0x152u:
      case 0x153u:
      case 0x15Au:
      case 0x163u:
      case 0x167u:
      case 0x168u:
      case 0x16Au:
      case 0x16Du:
      case 0x16Eu:
      case 0x171u:
      case 0x173u:
      case 0x174u:
      case 0x183u:
      case 0x184u:
      case 0x18Bu:
      case 0x18Cu:
      case 0x18Du:
      case 0x18Eu:
      case 0x18Fu:
      case 0x190u:
      case 0x191u:
        goto LABEL_27;
      case 2u:
      case 3u:
      case 4u:
      case 5u:
      case 6u:
      case 7u:
      case 8u:
      case 9u:
      case 0xAu:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x13u:
      case 0x16u:
      case 0x17u:
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x21u:
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
      case 0x36u:
      case 0x37u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x3Du:
      case 0x3Eu:
      case 0x40u:
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
      case 0x5Bu:
      case 0x5Cu:
      case 0x5Du:
      case 0x5Eu:
      case 0x5Fu:
      case 0x60u:
      case 0x62u:
      case 0x63u:
      case 0x64u:
      case 0x65u:
      case 0x6Eu:
      case 0x6Fu:
      case 0x70u:
      case 0x71u:
      case 0x74u:
      case 0x75u:
      case 0x76u:
      case 0x77u:
      case 0x78u:
      case 0x79u:
      case 0x7Au:
      case 0x7Bu:
      case 0x7Cu:
      case 0x7Du:
      case 0x7Eu:
      case 0x7Fu:
      case 0x81u:
      case 0x82u:
      case 0x86u:
      case 0x87u:
      case 0x88u:
      case 0x89u:
      case 0x8Au:
      case 0x8Bu:
      case 0x8Du:
      case 0x8Eu:
      case 0x8Fu:
      case 0x90u:
      case 0x91u:
      case 0x92u:
      case 0x93u:
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
      case 0xABu:
      case 0xB1u:
      case 0xB2u:
      case 0xB6u:
      case 0xB7u:
      case 0xB8u:
      case 0xBAu:
      case 0xBBu:
      case 0xBCu:
      case 0xBDu:
      case 0xBEu:
      case 0xBFu:
      case 0xC0u:
      case 0xC1u:
      case 0xC2u:
      case 0xC3u:
      case 0xC4u:
      case 0xC5u:
      case 0xC6u:
      case 0xC7u:
      case 0xC8u:
      case 0xC9u:
      case 0xCAu:
      case 0xCBu:
      case 0xCCu:
      case 0xCDu:
      case 0xD2u:
      case 0xD3u:
      case 0xD4u:
      case 0xD5u:
      case 0xD6u:
      case 0xD7u:
      case 0xD8u:
      case 0xD9u:
      case 0xDDu:
      case 0xDEu:
      case 0xDFu:
      case 0xE0u:
      case 0xE1u:
      case 0xE2u:
      case 0xE3u:
      case 0xE5u:
      case 0xE6u:
      case 0xE7u:
      case 0xE8u:
      case 0xE9u:
      case 0xEAu:
      case 0xECu:
      case 0xEEu:
      case 0xEFu:
      case 0xF0u:
      case 0xF1u:
      case 0xF2u:
      case 0xF3u:
      case 0xF4u:
      case 0xF5u:
      case 0xF7u:
      case 0xF9u:
      case 0xFBu:
      case 0xFCu:
      case 0xFDu:
      case 0xFEu:
      case 0xFFu:
      case 0x100u:
      case 0x101u:
      case 0x102u:
      case 0x103u:
      case 0x104u:
      case 0x105u:
      case 0x106u:
      case 0x107u:
      case 0x108u:
      case 0x109u:
      case 0x10Au:
      case 0x10Bu:
      case 0x10Cu:
      case 0x10Du:
      case 0x10Eu:
      case 0x10Fu:
      case 0x110u:
      case 0x111u:
      case 0x112u:
      case 0x113u:
      case 0x114u:
      case 0x115u:
      case 0x116u:
      case 0x117u:
      case 0x118u:
      case 0x119u:
      case 0x11Au:
      case 0x11Bu:
      case 0x11Eu:
      case 0x11Fu:
      case 0x120u:
      case 0x121u:
      case 0x122u:
      case 0x123u:
      case 0x124u:
      case 0x125u:
      case 0x126u:
      case 0x127u:
      case 0x128u:
      case 0x129u:
      case 0x12Au:
      case 0x12Bu:
      case 0x12Cu:
      case 0x12Du:
      case 0x12Eu:
      case 0x12Fu:
      case 0x130u:
      case 0x131u:
      case 0x132u:
      case 0x133u:
      case 0x13Au:
      case 0x13Bu:
      case 0x13Cu:
      case 0x13Du:
      case 0x13Eu:
      case 0x13Fu:
      case 0x140u:
      case 0x141u:
      case 0x142u:
      case 0x143u:
      case 0x144u:
      case 0x147u:
      case 0x148u:
      case 0x14Eu:
      case 0x150u:
      case 0x151u:
      case 0x154u:
      case 0x155u:
      case 0x156u:
      case 0x157u:
      case 0x158u:
      case 0x159u:
      case 0x15Bu:
      case 0x15Cu:
      case 0x15Du:
      case 0x15Eu:
      case 0x15Fu:
      case 0x160u:
      case 0x161u:
      case 0x162u:
      case 0x164u:
      case 0x165u:
      case 0x166u:
      case 0x169u:
      case 0x16Bu:
      case 0x16Cu:
      case 0x16Fu:
      case 0x170u:
      case 0x172u:
      case 0x175u:
      case 0x176u:
      case 0x177u:
      case 0x178u:
      case 0x179u:
      case 0x17Au:
      case 0x17Bu:
      case 0x17Cu:
      case 0x17Du:
      case 0x17Eu:
      case 0x17Fu:
      case 0x180u:
      case 0x181u:
      case 0x182u:
      case 0x185u:
      case 0x186u:
      case 0x187u:
      case 0x188u:
      case 0x189u:
      case 0x18Au:
        break;
      case 0x18u:
      case 0x19u:
      case 0x3Fu:
      case 0x58u:
      case 0x59u:
      case 0x5Au:
      case 0xADu:
      case 0xAEu:
      case 0xAFu:
      case 0xB0u:
      case 0xB3u:
      case 0xD1u:
      case 0xDAu:
      case 0xDBu:
      case 0xDCu:
      case 0xEBu:
      case 0xEDu:
      case 0xF6u:
      case 0xF8u:
      case 0x11Cu:
      case 0x11Du:
      case 0x145u:
      case 0x146u:
      case 0x14Fu:
        goto LABEL_32;
      default:
        if ( v5 > 0x2263 )
        {
          if ( v5 > 0x39DD )
          {
            if ( v5 <= 0x3BBF )
            {
              if ( v5 <= 0x3BB7 )
                break;
            }
            else
            {
              v7 = v5 - 15691;
              if ( (unsigned int)v7 > 0x2B )
                break;
              v8 = 0xC6000000033LL;
              if ( !_bittest64(&v8, v7) )
                break;
            }
            goto LABEL_32;
          }
          if ( v5 > 0x39D5 )
          {
LABEL_32:
            LOBYTE(v3) = 0;
            if ( !(unsigned __int8)sub_A73ED0(a1 + 72, 72) )
              return (unsigned int)sub_B49560(a1, 72) ^ 1;
            return v3;
          }
          if ( v5 <= 0x24FD )
          {
            if ( v5 > 0x24C1 )
            {
              v9 = 0xA00000000080021LL;
              if ( !_bittest64(&v9, v5 - 9410) )
                break;
            }
            else
            {
              if ( v5 > 0x2327 )
              {
                LOBYTE(v3) = 1;
                if ( v5 != 9259 && v5 - 9264 > 1 )
                  break;
                return v3;
              }
              if ( v5 <= 0x2302 )
                break;
              v6 = 0x100000000BLL;
              if ( !_bittest64(&v6, v5 - 8963) )
                break;
            }
          }
          else
          {
            if ( v5 <= 0x2545 )
            {
              if ( v5 <= 0x253A )
                break;
              LOBYTE(v3) = 1;
              if ( ((1LL << ((unsigned __int8)v5 - 59)) & 0x60D) == 0 )
                break;
              return v3;
            }
            if ( v5 - 14255 > 1 )
              break;
          }
          goto LABEL_27;
        }
        if ( v5 > 0x21C2 )
        {
          switch ( v5 )
          {
            case 0x21C3u:
            case 0x21C4u:
            case 0x2205u:
            case 0x2207u:
            case 0x220Bu:
            case 0x220Eu:
            case 0x2213u:
            case 0x2218u:
            case 0x221Du:
            case 0x2222u:
            case 0x2227u:
            case 0x222Cu:
            case 0x223Cu:
            case 0x223Eu:
            case 0x2242u:
            case 0x2245u:
            case 0x224Au:
            case 0x224Fu:
            case 0x2254u:
            case 0x2259u:
            case 0x225Eu:
            case 0x2263u:
              goto LABEL_27;
            default:
              goto LABEL_4;
          }
        }
        if ( v5 == 3185 )
          goto LABEL_27;
        if ( v5 <= 0xC71 )
        {
          if ( v5 == 2205 )
            goto LABEL_32;
          if ( v5 > 0x89D )
          {
            if ( v5 > 0x8A1 )
            {
              LOBYTE(v3) = 1;
              if ( v5 != 3147 && v5 != 3165 && v5 != 3071 )
                break;
              return v3;
            }
            if ( v5 <= 0x89F )
              break;
            goto LABEL_32;
          }
          if ( v5 == 2087 )
            goto LABEL_32;
          if ( v5 > 0x827 )
          {
            if ( v5 - 2089 > 3 )
              break;
            goto LABEL_32;
          }
          LOBYTE(v3) = 1;
          if ( v5 != 1248 )
            break;
          return v3;
        }
        if ( v5 <= 0x2076 )
        {
          if ( v5 <= 0x2060 )
          {
            if ( v5 != 3198 )
            {
              if ( v5 <= 0xC7D )
                break;
              LOBYTE(v3) = 1;
              if ( v5 > 0xD9B )
              {
                if ( v5 != 8170 )
                  break;
              }
              else if ( v5 <= 0xD97 && v5 - 3303 > 1 )
              {
                break;
              }
              return v3;
            }
            goto LABEL_32;
          }
          if ( ((1LL << ((unsigned __int8)v5 - 97)) & 0x280003) == 0 )
            break;
LABEL_27:
          LOBYTE(v3) = 1;
          return v3;
        }
        if ( v5 <= 0x2190 )
        {
          if ( v5 > 0x214B )
          {
            switch ( v5 )
            {
              case 0x214Cu:
              case 0x214Du:
              case 0x2151u:
              case 0x2162u:
              case 0x2163u:
              case 0x2164u:
              case 0x2165u:
              case 0x2166u:
              case 0x2167u:
              case 0x2168u:
              case 0x2169u:
              case 0x216Au:
              case 0x216Bu:
              case 0x216Cu:
              case 0x216Du:
              case 0x216Eu:
              case 0x216Fu:
              case 0x2170u:
              case 0x2171u:
              case 0x217Cu:
              case 0x217Du:
              case 0x217Eu:
              case 0x217Fu:
              case 0x2180u:
              case 0x2181u:
              case 0x2182u:
              case 0x2183u:
              case 0x2184u:
              case 0x2185u:
              case 0x2186u:
              case 0x2187u:
              case 0x2188u:
              case 0x2189u:
              case 0x218Au:
              case 0x218Bu:
              case 0x218Du:
              case 0x218Eu:
              case 0x2190u:
                goto LABEL_27;
              default:
                goto LABEL_4;
            }
          }
          LOBYTE(v3) = 1;
          if ( v5 == 8453 || v5 - 8466 <= 0xF )
            return v3;
        }
        break;
    }
  }
LABEL_4:
  LOBYTE(v3) = 0;
  return v3;
}
