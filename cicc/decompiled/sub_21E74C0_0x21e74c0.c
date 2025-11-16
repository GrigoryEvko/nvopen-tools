// Function: sub_21E74C0
// Address: 0x21e74c0
//
char __fastcall sub_21E74C0(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  __int64 v4; // rdx
  _WORD *v5; // rdx
  _WORD *v6; // rdx
  char *v7; // rdx
  char *v8; // rax
  bool v9; // cf
  char *v10; // rdx
  char *v11; // rax
  char *v12; // rdx
  char *v13; // rax
  __int64 v14; // rdx
  _DWORD *v15; // rdx
  _DWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // r15
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  int v24; // ecx
  bool v25; // zf
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  int v30; // edi
  _QWORD *v31; // rdx
  _QWORD *v32; // rdx
  __int64 v33; // rdx
  _QWORD *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  _QWORD *v38; // rdx
  _QWORD *v39; // rdx
  __int64 v40; // rdx
  _QWORD *v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rdx
  _QWORD *v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rdx

  v22 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( !strcmp(a4, "mid") )
  {
    v23 = (v22 >> 1) & 7;
    if ( (_BYTE)v23 == 2 )
    {
      v22 = *(_QWORD *)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) == v22 )
      {
        LOBYTE(v22) = sub_16E7EE0(a3, (char *)"c", 1u);
      }
      else
      {
        *(_BYTE *)v22 = 99;
        ++*(_QWORD *)(a3 + 24);
      }
    }
    else if ( (unsigned __int8)v23 > 2u )
    {
      v22 = *(_QWORD *)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) == v22 )
      {
        LOBYTE(v22) = sub_16E7EE0(a3, "d", 1u);
      }
      else
      {
        *(_BYTE *)v22 = 100;
        ++*(_QWORD *)(a3 + 24);
      }
    }
    else if ( (_BYTE)v23 )
    {
      v22 = *(_QWORD *)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) == v22 )
      {
        LOBYTE(v22) = sub_16E7EE0(a3, "b", 1u);
      }
      else
      {
        *(_BYTE *)v22 = 98;
        ++*(_QWORD *)(a3 + 24);
      }
    }
    else
    {
      v22 = *(_QWORD *)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) == v22 )
      {
        LOBYTE(v22) = sub_16E7EE0(a3, "a", 1u);
      }
      else
      {
        *(_BYTE *)v22 = 97;
        ++*(_QWORD *)(a3 + 24);
      }
    }
    return v22;
  }
  if ( !strcmp(a4, "rowcol") )
  {
    v25 = (v22 & 1) == 0;
    v26 = *(_QWORD *)(a3 + 24);
    v27 = *(_QWORD *)(a3 + 16);
    if ( v25 )
      goto LABEL_14;
    goto LABEL_32;
  }
  if ( !strcmp(a4, "shape") )
  {
    switch ( BYTE4(v22) )
    {
      case 1:
        v44 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v44) <= 5 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n8k4", 6u);
        }
        else
        {
          *(_DWORD *)v44 = 946747501;
          *(_WORD *)(v44 + 4) = 13419;
          *(_QWORD *)(a3 + 24) += 6LL;
          LOBYTE(v22) = 107;
        }
        break;
      case 2:
        v42 = *(_QWORD *)(a3 + 24);
        v22 = *(_QWORD *)(a3 + 16) - v42;
        if ( v22 <= 6 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n8k16", 7u);
        }
        else
        {
          *(_DWORD *)v42 = 946747501;
          *(_WORD *)(v42 + 4) = 12651;
          *(_BYTE *)(v42 + 6) = 54;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
        break;
      case 3:
        v46 = *(_QWORD *)(a3 + 24);
        v22 = *(_QWORD *)(a3 + 16) - v46;
        if ( v22 <= 6 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n8k32", 7u);
        }
        else
        {
          *(_DWORD *)v46 = 946747501;
          *(_WORD *)(v46 + 4) = 13163;
          *(_BYTE *)(v46 + 6) = 50;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
        break;
      case 4:
        v43 = *(_QWORD *)(a3 + 24);
        v22 = *(_QWORD *)(a3 + 16) - v43;
        if ( v22 <= 6 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n8k64", 7u);
        }
        else
        {
          *(_DWORD *)v43 = 946747501;
          *(_WORD *)(v43 + 4) = 13931;
          *(_BYTE *)(v43 + 6) = 52;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
        break;
      case 5:
        v45 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v45 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n8k128", 8u);
        }
        else
        {
          *v45 = 0x3832316B386E386DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 6:
        v41 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v41 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m8n32k16", 8u);
        }
        else
        {
          *v41 = 0x36316B32336E386DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x10:
        v40 = *(_QWORD *)(a3 + 24);
        v22 = *(_QWORD *)(a3 + 16) - v40;
        if ( v22 <= 6 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k4", 7u);
        }
        else
        {
          *(_DWORD *)v40 = 1849045357;
          *(_WORD *)(v40 + 4) = 27448;
          *(_BYTE *)(v40 + 6) = 52;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
        break;
      case 0x11:
        v36 = *(_QWORD *)(a3 + 24);
        v22 = *(_QWORD *)(a3 + 16) - v36;
        if ( v22 <= 6 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k8", 7u);
        }
        else
        {
          *(_DWORD *)v36 = 1849045357;
          *(_WORD *)(v36 + 4) = 27448;
          *(_BYTE *)(v36 + 6) = 56;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
        break;
      case 0x12:
        v38 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v38 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k16", 8u);
        }
        else
        {
          *v38 = 0x36316B386E36316DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x13:
        v34 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v34 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k32", 8u);
        }
        else
        {
          *v34 = 0x32336B386E36316DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x14:
        v39 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v39 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k64", 8u);
        }
        else
        {
          *v39 = 0x34366B386E36316DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x15:
        v35 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v35) <= 8 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k128", 9u);
        }
        else
        {
          *(_BYTE *)(v35 + 8) = 56;
          *(_QWORD *)v35 = 0x32316B386E36316DLL;
          *(_QWORD *)(a3 + 24) += 9LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x16:
        v37 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v37) <= 8 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n8k256", 9u);
        }
        else
        {
          *(_BYTE *)(v37 + 8) = 54;
          *(_QWORD *)v37 = 0x35326B386E36316DLL;
          *(_QWORD *)(a3 + 24) += 9LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x17:
        v33 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v33) <= 8 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n16k16", 9u);
        }
        else
        {
          *(_BYTE *)(v33 + 8) = 54;
          *(_QWORD *)v33 = 0x316B36316E36316DLL;
          *(_QWORD *)(a3 + 24) += 9LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x18:
        v32 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v32 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m32n8k16", 8u);
        }
        else
        {
          *v32 = 0x36316B386E32336DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      case 0x19:
        v31 = *(_QWORD **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v31 <= 7u )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "m16n16k8", 8u);
        }
        else
        {
          *v31 = 0x386B36316E36316DLL;
          *(_QWORD *)(a3 + 24) += 8LL;
          LOBYTE(v22) = 109;
        }
        break;
      default:
        ++*(_DWORD *)(v19 + 72);
        BUG();
    }
  }
  else
  {
    if ( !strcmp(a4, "ety") )
    {
      v29 = a3;
      v30 = (unsigned __int8)(v22 >> 4);
    }
    else
    {
      v24 = *(unsigned __int8 *)a4;
      if ( v24 == 97 && a4[1] == 108 && !a4[2] )
      {
        v25 = (v22 & 0x3000000) == 0;
        v26 = *(_QWORD *)(a3 + 24);
        v27 = *(_QWORD *)(a3 + 16);
        if ( v25 )
          goto LABEL_14;
        goto LABEL_32;
      }
      if ( v24 == 98 && a4[1] == 108 && !a4[2] )
      {
        v25 = (v22 & 0xC000000) == 0;
        v26 = *(_QWORD *)(a3 + 24);
        v27 = *(_QWORD *)(a3 + 16);
        if ( v25 )
        {
LABEL_14:
          v22 = v27 - v26;
          if ( v22 <= 2 )
          {
            LOBYTE(v22) = sub_16E7EE0(a3, "row", 3u);
          }
          else
          {
            *(_BYTE *)(v26 + 2) = 119;
            *(_WORD *)v26 = 28530;
            *(_QWORD *)(a3 + 24) += 3LL;
          }
          return v22;
        }
LABEL_32:
        if ( (unsigned __int64)(v27 - v26) <= 2 )
        {
          LOBYTE(v22) = sub_16E7EE0(a3, "col", 3u);
        }
        else
        {
          *(_BYTE *)(v26 + 2) = 108;
          *(_WORD *)v26 = 28515;
          *(_QWORD *)(a3 + 24) += 3LL;
          LOBYTE(v22) = 99;
        }
        return v22;
      }
      if ( !strcmp(a4, "aty") )
      {
        v30 = BYTE1(v22);
        v29 = a3;
      }
      else
      {
        if ( strcmp(a4, "bty") )
        {
          if ( !strcmp(a4, "opc") )
          {
            LOBYTE(v22) = (unsigned __int8)v22 >> 4;
            if ( (_BYTE)v22 == 1 )
            {
              v47 = *(_QWORD *)(a3 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v47) <= 8 )
              {
                LOBYTE(v22) = sub_16E7EE0(a3, ".and.popc", 9u);
              }
              else
              {
                *(_BYTE *)(v47 + 8) = 99;
                *(_QWORD *)v47 = 0x706F702E646E612ELL;
                *(_QWORD *)(a3 + 24) += 9LL;
                LOBYTE(v22) = 46;
              }
            }
            else if ( (_BYTE)v22 == 2 )
            {
              v28 = *(_QWORD *)(a3 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v28) <= 8 )
              {
                LOBYTE(v22) = sub_16E7EE0(a3, ".xor.popc", 9u);
              }
              else
              {
                *(_BYTE *)(v28 + 8) = 99;
                *(_QWORD *)v28 = 0x706F702E726F782ELL;
                *(_QWORD *)(a3 + 24) += 9LL;
                LOBYTE(v22) = 46;
              }
            }
          }
          else if ( !strcmp(a4, "rnd") )
          {
            LOBYTE(v22) = v22 & 7;
            if ( (_BYTE)v22 == 3 )
            {
              v22 = *(_QWORD *)(a3 + 24);
              if ( *(_QWORD *)(a3 + 16) - v22 <= 2 )
              {
                LOBYTE(v22) = sub_16E7EE0(a3, ".rp", 3u);
              }
              else
              {
                *(_WORD *)v22 = 29230;
                *(_BYTE *)(v22 + 2) = 112;
                *(_QWORD *)(a3 + 24) += 3LL;
              }
            }
            else if ( (unsigned __int8)v22 > 3u )
            {
              if ( (_BYTE)v22 == 4 )
              {
                v22 = *(_QWORD *)(a3 + 24);
                if ( *(_QWORD *)(a3 + 16) - v22 <= 2 )
                {
                  LOBYTE(v22) = sub_16E7EE0(a3, ".rz", 3u);
                }
                else
                {
                  *(_WORD *)v22 = 29230;
                  *(_BYTE *)(v22 + 2) = 122;
                  *(_QWORD *)(a3 + 24) += 3LL;
                }
              }
            }
            else if ( (_BYTE)v22 == 1 )
            {
              v22 = *(_QWORD *)(a3 + 24);
              if ( *(_QWORD *)(a3 + 16) - v22 <= 2 )
              {
                LOBYTE(v22) = sub_16E7EE0(a3, ".rn", 3u);
              }
              else
              {
                *(_WORD *)v22 = 29230;
                *(_BYTE *)(v22 + 2) = 110;
                *(_QWORD *)(a3 + 24) += 3LL;
              }
            }
            else if ( (_BYTE)v22 == 2 )
            {
              v22 = *(_QWORD *)(a3 + 24);
              if ( *(_QWORD *)(a3 + 16) - v22 <= 2 )
              {
                LOBYTE(v22) = sub_16E7EE0(a3, ".rm", 3u);
              }
              else
              {
                *(_WORD *)v22 = 29230;
                *(_BYTE *)(v22 + 2) = 109;
                *(_QWORD *)(a3 + 24) += 3LL;
              }
            }
          }
          else if ( !strcmp(a4, "satf") && (v22 & 0x10000000) != 0 )
          {
            v22 = *(_QWORD *)(a3 + 24);
            if ( *(_QWORD *)(a3 + 16) - v22 <= 9 )
            {
              LOBYTE(v22) = sub_16E7EE0(a3, ".satfinite", 0xAu);
            }
            else
            {
              qmemcpy((void *)v22, ".satfinite", 10);
              *(_QWORD *)(a3 + 24) += 10LL;
            }
          }
          return v22;
        }
        v29 = a3;
        v30 = BYTE2(v22);
      }
    }
    switch ( v30 )
    {
      case 1:
        v5 = *(_WORD **)(v29 + 24);
        if ( *(_QWORD *)(v29 + 16) - (_QWORD)v5 <= 1u )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "b1", 2u);
        }
        else
        {
          *v5 = 12642;
          *(_QWORD *)(v29 + 24) += 2LL;
          LOBYTE(v22) = 98;
        }
        break;
      case 2:
        v6 = *(_WORD **)(v29 + 24);
        if ( *(_QWORD *)(v29 + 16) - (_QWORD)v6 <= 1u )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "s4", 2u);
        }
        else
        {
          *v6 = 13427;
          *(_QWORD *)(v29 + 24) += 2LL;
          LOBYTE(v22) = 115;
        }
        break;
      case 3:
        v7 = *(char **)(v29 + 24);
        v8 = *(char **)(v29 + 16);
        v9 = v8 == v7;
        v22 = v8 - v7;
        if ( v9 || v22 == 1 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "u4", 2u);
        }
        else
        {
          *(_WORD *)v7 = 13429;
          *(_QWORD *)(v29 + 24) += 2LL;
        }
        break;
      case 4:
        v10 = *(char **)(v29 + 24);
        v11 = *(char **)(v29 + 16);
        v9 = v11 == v10;
        v22 = v11 - v10;
        if ( v9 || v22 == 1 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "s8", 2u);
        }
        else
        {
          *(_WORD *)v10 = 14451;
          *(_QWORD *)(v29 + 24) += 2LL;
        }
        break;
      case 5:
        v12 = *(char **)(v29 + 24);
        v13 = *(char **)(v29 + 16);
        v9 = v13 == v12;
        v22 = v13 - v12;
        if ( v9 || v22 == 1 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "u8", 2u);
        }
        else
        {
          *(_WORD *)v12 = 14453;
          *(_QWORD *)(v29 + 24) += 2LL;
        }
        break;
      case 6:
        v14 = *(_QWORD *)(v29 + 24);
        v22 = *(_QWORD *)(v29 + 16) - v14;
        if ( v22 <= 2 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "f16", 3u);
        }
        else
        {
          *(_BYTE *)(v14 + 2) = 54;
          *(_WORD *)v14 = 12646;
          *(_QWORD *)(v29 + 24) += 3LL;
        }
        break;
      case 7:
        v15 = *(_DWORD **)(v29 + 24);
        v22 = *(_QWORD *)(v29 + 16) - (_QWORD)v15;
        if ( v22 <= 3 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "bf16", 4u);
        }
        else
        {
          *v15 = 909207138;
          *(_QWORD *)(v29 + 24) += 4LL;
        }
        break;
      case 8:
        v16 = *(_DWORD **)(v29 + 24);
        v22 = *(_QWORD *)(v29 + 16) - (_QWORD)v16;
        if ( v22 <= 3 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "tf32", 4u);
        }
        else
        {
          *v16 = 842229364;
          *(_QWORD *)(v29 + 24) += 4LL;
        }
        break;
      case 9:
        v17 = *(_QWORD *)(v29 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v29 + 16) - v17) <= 2 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "f64", 3u);
        }
        else
        {
          *(_BYTE *)(v17 + 2) = 52;
          *(_WORD *)v17 = 13926;
          *(_QWORD *)(v29 + 24) += 3LL;
          LOBYTE(v22) = 102;
        }
        break;
      case 10:
        v18 = *(_QWORD *)(v29 + 24);
        v22 = *(_QWORD *)(v29 + 16) - v18;
        if ( v22 <= 2 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "f32", 3u);
        }
        else
        {
          *(_BYTE *)(v18 + 2) = 50;
          *(_WORD *)v18 = 13158;
          *(_QWORD *)(v29 + 24) += 3LL;
        }
        break;
      case 11:
        v4 = *(_QWORD *)(v29 + 24);
        v22 = *(_QWORD *)(v29 + 16) - v4;
        if ( v22 <= 2 )
        {
          LOBYTE(v22) = sub_16E7EE0(v29, "s32", 3u);
        }
        else
        {
          *(_BYTE *)(v4 + 2) = 50;
          *(_WORD *)v4 = 13171;
          *(_QWORD *)(v29 + 24) += 3LL;
        }
        break;
      default:
        sub_16BD130("Wrong MMA element type", 1u);
    }
  }
  return v22;
}
