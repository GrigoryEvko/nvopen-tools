// Function: sub_989640
// Address: 0x989640
//
__int64 __fastcall sub_989640(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned int a5, char a6)
{
  int v9; // r15d
  __int64 v10; // r9
  unsigned int v11; // r15d
  char v12; // r14
  int v13; // eax
  int v14; // edx
  char v15; // dl
  __int64 v16; // rax
  unsigned int v17; // eax
  int v18; // ebx
  int v19; // ebx
  int v20; // r15d
  int v21; // eax
  int v22; // ebx
  __int64 v23; // rdi
  __int64 v24; // rsi
  _BYTE *v25; // rax
  bool v27; // zf
  unsigned int v28; // ebx
  unsigned int v29; // r8d
  int v30; // eax
  bool v31; // dl
  int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  unsigned int v40; // [rsp+18h] [rbp-38h]
  __int64 v41; // [rsp+18h] [rbp-38h]
  __int64 v42; // [rsp+18h] [rbp-38h]
  unsigned int v43; // [rsp+18h] [rbp-38h]

  if ( a2 == 15 )
  {
    *(_QWORD *)(a1 + 8) = a4;
    *(_QWORD *)a1 = 0x3FF00000000LL;
    return a1;
  }
  if ( !a2 )
  {
    *(_QWORD *)a1 = 1023;
    *(_QWORD *)(a1 + 8) = a4;
    return a1;
  }
  v9 = a5 & 0x3FC;
  if ( (a5 & 0x3FC) != 0 )
  {
    if ( a2 == 7 )
    {
      *(_QWORD *)(a1 + 8) = a4;
      *(_QWORD *)a1 = 0x3FC00000003LL;
      return a1;
    }
    if ( a2 == 8 )
    {
      *(_QWORD *)(a1 + 8) = a4;
      *(_QWORD *)a1 = 0x3000003FCLL;
      return a1;
    }
    if ( a6
      && *(_BYTE *)a4 == 85
      && (v16 = *(_QWORD *)(a4 - 32)) != 0
      && !*(_BYTE *)v16
      && *(_QWORD *)(v16 + 24) == *(_QWORD *)(a4 + 80)
      && *(_DWORD *)(v16 + 36) == 170
      && (v12 = a6, (v34 = *(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF))) != 0) )
    {
      v35 = a4;
      v36 = a3;
      v40 = a5;
      v17 = sub_C650C0(a5);
      a5 = v40;
      a3 = v36;
      a4 = v35;
      v10 = v34;
      v11 = v17;
    }
    else
    {
      v10 = a4;
      v11 = a5;
      v12 = 0;
    }
    if ( (a5 & 0xFFFFFF9F) != 0 )
    {
      v13 = a5 & 0x3C;
      if ( (a5 & 0xFFFFFDFB) != 0 )
      {
        switch ( a2 )
        {
          case 1u:
            *(_DWORD *)a1 = 1023;
            *(_DWORD *)(a1 + 4) = v11;
            *(_QWORD *)(a1 + 8) = v10;
            return a1;
          case 9u:
            *(_DWORD *)a1 = 1020;
            *(_DWORD *)(a1 + 4) = v11 | 3;
            *(_QWORD *)(a1 + 8) = v10;
            return a1;
          case 6u:
            *(_QWORD *)(a1 + 8) = v10;
            *(_DWORD *)a1 = v11 | 3;
            *(_DWORD *)(a1 + 4) = 1020;
            return a1;
          case 0xEu:
            *(_DWORD *)a1 = v11;
            *(_DWORD *)(a1 + 4) = 1023;
            *(_QWORD *)(a1 + 8) = v10;
            return a1;
        }
        v14 = a5 & 0x90;
        if ( v13 == a5 )
        {
          if ( v12 )
          {
            if ( a2 > 0xB )
            {
              if ( a2 - 12 <= 1 )
              {
                *(_QWORD *)(a1 + 8) = v10;
                *(_QWORD *)a1 = 0x3000003FCLL;
                return a1;
              }
            }
            else
            {
              if ( a2 > 9 )
              {
                *(_QWORD *)(a1 + 8) = v10;
                *(_QWORD *)a1 = 0x3FF00000000LL;
                return a1;
              }
              if ( a2 <= 3 )
              {
                *(_QWORD *)(a1 + 8) = v10;
                *(_QWORD *)a1 = 0x3FC00000003LL;
                return a1;
              }
              if ( a2 - 4 <= 1 )
              {
                *(_QWORD *)a1 = 1023;
                *(_QWORD *)(a1 + 8) = v10;
                return a1;
              }
            }
          }
          else
          {
            v27 = v14 == a5;
            v30 = 28;
            v31 = v14 != a5;
            if ( !v27 )
              v30 = 12;
            v32 = 8 * v31 + 1008;
            if ( a2 > 0xB )
            {
              if ( a2 - 12 <= 1 )
              {
                *(_QWORD *)(a1 + 8) = v10;
                *(_DWORD *)(a1 + 4) = v30 | 3;
                *(_DWORD *)a1 = v30 ^ 0x3FC | v11;
                return a1;
              }
            }
            else
            {
              if ( a2 > 9 )
              {
                *(_QWORD *)(a1 + 8) = v10;
                *(_DWORD *)(a1 + 4) = v32 | 3;
                *(_DWORD *)a1 = v32 ^ 0x3FC | v11;
                return a1;
              }
              if ( a2 <= 3 )
              {
                *(_DWORD *)(a1 + 4) = v32;
                *(_QWORD *)(a1 + 8) = v10;
                *(_DWORD *)a1 = v32 ^ 0x3FF | v11;
                return a1;
              }
              if ( a2 - 4 <= 1 )
              {
                *(_DWORD *)(a1 + 4) = v30;
                *(_QWORD *)(a1 + 8) = v10;
                *(_DWORD *)a1 = v30 ^ 0x3FF | v11;
                return a1;
              }
            }
          }
          goto LABEL_18;
        }
        if ( (a5 & 0xFFFFFC3F) != 0 )
        {
LABEL_18:
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)a1 = 0x3FF000003FFLL;
          return a1;
        }
        v27 = v14 == a5;
        v28 = 252;
        v29 = 896;
        if ( !v27 )
        {
          v28 = 508;
          v29 = 768;
        }
        if ( v12 )
        {
          v38 = v10;
          v43 = sub_C650C0(v29);
          v33 = sub_C650C0(v28);
          v10 = v38;
          v29 = v43;
          v28 = v33;
        }
        if ( a2 > 0xB )
        {
          v28 |= 3u;
          if ( a2 - 12 > 1 )
            goto LABEL_18;
        }
        else
        {
          if ( a2 > 9 )
          {
            v29 |= 3u;
            goto LABEL_90;
          }
          if ( a2 <= 3 )
          {
LABEL_90:
            *(_DWORD *)(a1 + 4) = v29;
            *(_QWORD *)(a1 + 8) = v10;
            *(_DWORD *)a1 = ~(_WORD)v29 & 0x3FF | v11;
            return a1;
          }
          if ( a2 - 4 > 1 )
            goto LABEL_18;
        }
        *(_DWORD *)(a1 + 4) = v28;
        *(_QWORD *)(a1 + 8) = v10;
        *(_DWORD *)a1 = ~(_WORD)v28 & 0x3FF | v11;
        return a1;
      }
      switch ( a2 )
      {
        case 1u:
        case 0xEu:
          v22 = -(v12 == 0);
          if ( v13 == a5 )
            goto LABEL_41;
          goto LABEL_46;
        case 2u:
        case 0xDu:
          v19 = 1023;
          v20 = 0;
          if ( v13 != a5 )
            break;
          v18 = -(v12 == 0);
LABEL_49:
          v19 = (v18 & 4) + 3;
          v20 = v12 == 0 ? 1016 : 1020;
          break;
        case 3u:
        case 0xCu:
          if ( v13 == a5 )
          {
            v19 = 3;
            v20 = 1020;
          }
          else
          {
            v22 = -(v12 == 0);
LABEL_46:
            v19 = (v22 & 4) + 507;
            v20 = v12 == 0 ? 512 : 516;
          }
          break;
        case 4u:
        case 0xBu:
          if ( v13 == a5 )
          {
            v19 = 1023;
            v20 = 0;
          }
          else
          {
            v18 = -(v12 == 0);
LABEL_35:
            v19 = (v18 & 0xFFFFFFFC) + 519;
            v20 = v12 == 0 ? 508 : 504;
          }
          break;
        case 5u:
        case 0xAu:
          v19 = 3;
          v20 = 1020;
          if ( v13 != a5 )
            break;
          v22 = -(v12 == 0);
LABEL_41:
          v19 = (v22 & 0xFFFFFFFC) + 1023;
          v20 = v12 == 0 ? 4 : 0;
          break;
        case 6u:
        case 9u:
          v18 = -(v12 == 0);
          if ( v13 != a5 )
            goto LABEL_35;
          goto LABEL_49;
        default:
          goto LABEL_111;
      }
      v41 = v10;
      if ( (unsigned __int8)sub_B535C0(a2, a2, a3, a4) )
      {
        v21 = v20;
        v20 = v19;
        v19 = v21;
      }
      *(_DWORD *)a1 = v19;
      *(_DWORD *)(a1 + 4) = v20;
      *(_QWORD *)(a1 + 8) = v41;
    }
    else
    {
      v23 = *(_QWORD *)(a4 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
        v23 = **(_QWORD **)(v23 + 16);
      v42 = v10;
      v37 = a3;
      v24 = sub_BCAC60(v23);
      if ( (unsigned __int16)sub_B2DB90(v37, v24) >> 8 )
        goto LABEL_18;
      v25 = (_BYTE *)sub_C94E20(qword_4F863F0);
      if ( v25 ? *v25 : LOBYTE(qword_4F863F0[2]) )
        goto LABEL_18;
      switch ( a2 )
      {
        case 1u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x600000039FLL;
          break;
        case 2u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x3800000007FLL;
          break;
        case 3u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x3E00000001FLL;
          break;
        case 4u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x1C000003E3LL;
          break;
        case 5u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x7C00000383LL;
          break;
        case 6u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x39C00000063LL;
          break;
        case 9u:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x630000039CLL;
          break;
        case 0xAu:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x3830000007CLL;
          break;
        case 0xBu:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x3E30000001CLL;
          break;
        case 0xCu:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x1F000003E0LL;
          break;
        case 0xDu:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x7F00000380LL;
          break;
        case 0xEu:
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)a1 = 0x39F00000060LL;
          break;
        default:
LABEL_111:
          BUG();
      }
    }
  }
  else
  {
    v39 = a4;
    v15 = sub_B535B0(a2);
    *(_QWORD *)(a1 + 8) = v39;
    if ( !v15 )
      v9 = 1023;
    *(_DWORD *)a1 = v15 != 0 ? 0x3FF : 0;
    *(_DWORD *)(a1 + 4) = v9;
  }
  return a1;
}
