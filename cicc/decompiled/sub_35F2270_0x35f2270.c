// Function: sub_35F2270
// Address: 0x35f2270
//
char __fastcall sub_35F2270(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  __int64 v8; // rbx
  _BYTE *v9; // rax
  int v10; // eax
  int v11; // ebx
  size_t v12; // rdx
  char *v13; // rsi
  int v14; // edi
  bool v15; // zf
  char v16; // bl
  __int64 v17; // rdx
  unsigned __int64 v18; // rbx
  __int64 v19; // rbx
  _QWORD *v20; // rdx
  _QWORD *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // rdx
  _QWORD *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  _QWORD *v31; // rdx
  __int64 v32; // rdx
  _QWORD *v33; // rdx
  _QWORD *v34; // rdx
  __int64 v35; // rdx
  const char *v36; // rsi

  v6 = *(_QWORD *)(a2 + 16) + 16LL * a3;
  v7 = *(_QWORD *)(v6 + 8);
  if ( !strcmp(a5, "mid") )
  {
    v8 = (v7 >> 1) & 7;
    if ( (_BYTE)v8 == 2 )
    {
      v9 = *(_BYTE **)(a4 + 32);
      v12 = 1;
      v13 = (char *)"c";
      if ( *(_BYTE **)(a4 + 24) != v9 )
      {
        *v9 = 99;
        ++*(_QWORD *)(a4 + 32);
        return (char)v9;
      }
      goto LABEL_22;
    }
    if ( (unsigned __int8)v8 <= 2u )
    {
      if ( (_BYTE)v8 )
      {
        v9 = *(_BYTE **)(a4 + 32);
        if ( *(_BYTE **)(a4 + 24) != v9 )
        {
          *v9 = 98;
          ++*(_QWORD *)(a4 + 32);
          return (char)v9;
        }
        v12 = 1;
        v13 = "b";
      }
      else
      {
        v9 = *(_BYTE **)(a4 + 32);
        v12 = 1;
        v13 = "a";
        if ( *(_BYTE **)(a4 + 24) != v9 )
        {
          *v9 = 97;
          ++*(_QWORD *)(a4 + 32);
          return (char)v9;
        }
      }
      goto LABEL_22;
    }
    if ( (_BYTE)v8 == 3 )
    {
      v9 = *(_BYTE **)(a4 + 32);
      v12 = 1;
      v13 = "d";
      if ( *(_BYTE **)(a4 + 24) != v9 )
      {
        *v9 = 100;
        ++*(_QWORD *)(a4 + 32);
        return (char)v9;
      }
LABEL_22:
      LOBYTE(v9) = sub_CB6200(a4, (unsigned __int8 *)v13, v12);
      return (char)v9;
    }
LABEL_128:
    BUG();
  }
  if ( !strcmp(a5, "rowcol") )
  {
    v17 = *(_QWORD *)(a4 + 32);
    v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v17);
    if ( (v7 & 1) != 0 )
    {
LABEL_36:
      if ( (unsigned __int64)v9 > 2 )
      {
        *(_BYTE *)(v17 + 2) = 108;
        *(_WORD *)v17 = 28515;
        *(_QWORD *)(a4 + 32) += 3LL;
        LOBYTE(v9) = 99;
        return (char)v9;
      }
      v12 = 3;
      v13 = "col";
      goto LABEL_22;
    }
LABEL_39:
    if ( (unsigned __int64)v9 > 2 )
    {
      *(_BYTE *)(v17 + 2) = 119;
      *(_WORD *)v17 = 28530;
      *(_QWORD *)(a4 + 32) += 3LL;
      return (char)v9;
    }
    v12 = 3;
    v13 = "row";
    goto LABEL_22;
  }
  if ( !strcmp(a5, "shape") )
  {
    switch ( BYTE4(v7) )
    {
      case 1:
        v35 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v35);
        if ( (unsigned __int64)v9 > 5 )
        {
          *(_DWORD *)v35 = 946747501;
          *(_WORD *)(v35 + 4) = 13419;
          *(_QWORD *)(a4 + 32) += 6LL;
          return (char)v9;
        }
        v12 = 6;
        v13 = "m8n8k4";
        goto LABEL_22;
      case 2:
        v28 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v28);
        if ( (unsigned __int64)v9 > 6 )
        {
          *(_DWORD *)v28 = 946747501;
          *(_WORD *)(v28 + 4) = 12651;
          *(_BYTE *)(v28 + 6) = 54;
          *(_QWORD *)(a4 + 32) += 7LL;
          return (char)v9;
        }
        v12 = 7;
        v13 = "m8n8k16";
        goto LABEL_22;
      case 3:
        v32 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v32);
        if ( (unsigned __int64)v9 > 6 )
        {
          *(_DWORD *)v32 = 946747501;
          *(_WORD *)(v32 + 4) = 13163;
          *(_BYTE *)(v32 + 6) = 50;
          *(_QWORD *)(a4 + 32) += 7LL;
          return (char)v9;
        }
        v12 = 7;
        v13 = "m8n8k32";
        goto LABEL_22;
      case 4:
        v24 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v24);
        if ( (unsigned __int64)v9 > 6 )
        {
          *(_DWORD *)v24 = 946747501;
          *(_WORD *)(v24 + 4) = 13931;
          *(_BYTE *)(v24 + 6) = 52;
          *(_QWORD *)(a4 + 32) += 7LL;
          return (char)v9;
        }
        v12 = 7;
        v13 = "m8n8k64";
        goto LABEL_22;
      case 5:
        v34 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v34 > 7u )
        {
          *v34 = 0x3832316B386E386DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m8n8k128";
        goto LABEL_22;
      case 6:
        v26 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v26 > 7u )
        {
          *v26 = 0x36316B32336E386DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m8n32k16";
        goto LABEL_22;
      case 0x10:
        v30 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v30);
        if ( (unsigned __int64)v9 > 6 )
        {
          *(_DWORD *)v30 = 1849045357;
          *(_WORD *)(v30 + 4) = 27448;
          *(_BYTE *)(v30 + 6) = 52;
          *(_QWORD *)(a4 + 32) += 7LL;
          return (char)v9;
        }
        v12 = 7;
        v13 = "m16n8k4";
        goto LABEL_22;
      case 0x11:
        v22 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v22);
        if ( (unsigned __int64)v9 > 6 )
        {
          *(_DWORD *)v22 = 1849045357;
          *(_WORD *)(v22 + 4) = 27448;
          *(_BYTE *)(v22 + 6) = 56;
          *(_QWORD *)(a4 + 32) += 7LL;
          return (char)v9;
        }
        v12 = 7;
        v13 = "m16n8k8";
        goto LABEL_22;
      case 0x12:
        v33 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v33 > 7u )
        {
          *v33 = 0x36316B386E36316DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m16n8k16";
        goto LABEL_22;
      case 0x13:
        v27 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v27 > 7u )
        {
          *v27 = 0x32336B386E36316DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m16n8k32";
        goto LABEL_22;
      case 0x14:
        v31 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v31 > 7u )
        {
          *v31 = 0x34366B386E36316DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m16n8k64";
        goto LABEL_22;
      case 0x15:
        v23 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v23) > 8 )
        {
          *(_BYTE *)(v23 + 8) = 56;
          *(_QWORD *)v23 = 0x32316B386E36316DLL;
          *(_QWORD *)(a4 + 32) += 9LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 9;
        v13 = "m16n8k128";
        goto LABEL_22;
      case 0x16:
        v29 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v29) > 8 )
        {
          *(_BYTE *)(v29 + 8) = 54;
          *(_QWORD *)v29 = 0x35326B386E36316DLL;
          *(_QWORD *)(a4 + 32) += 9LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 9;
        v13 = "m16n8k256";
        goto LABEL_22;
      case 0x17:
        v25 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v25) > 8 )
        {
          *(_BYTE *)(v25 + 8) = 54;
          *(_QWORD *)v25 = 0x316B36316E36316DLL;
          *(_QWORD *)(a4 + 32) += 9LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 9;
        v13 = "m16n16k16";
        goto LABEL_22;
      case 0x18:
        v21 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v21 > 7u )
        {
          *v21 = 0x36316B386E32336DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m32n8k16";
        goto LABEL_22;
      case 0x19:
        v20 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v20 > 7u )
        {
          *v20 = 0x386B36316E36316DLL;
          *(_QWORD *)(a4 + 32) += 8LL;
          LOBYTE(v9) = 109;
          return (char)v9;
        }
        v12 = 8;
        v13 = "m16n16k8";
        break;
      default:
        goto LABEL_128;
    }
    goto LABEL_22;
  }
  if ( !strcmp(a5, "ety") )
  {
    v18 = v7 >> 4;
LABEL_44:
    v14 = (unsigned __int8)v18;
    goto LABEL_45;
  }
  v10 = *(unsigned __int8 *)a5;
  if ( v10 == 97 && a5[1] == 108 && !a5[2] )
  {
    v11 = (v7 >> 24) & 3;
    if ( v11 && (_BYTE)v11 != 1 )
    {
      if ( (_BYTE)v11 != 2 )
        goto LABEL_128;
      goto LABEL_52;
    }
    goto LABEL_38;
  }
  if ( v10 == 98 && a5[1] == 108 && !a5[2] )
  {
    if ( (v7 & 0xC000000) != 0 )
    {
      v19 = (v7 >> 26) & 3;
      if ( (_BYTE)v19 == 1 )
      {
LABEL_38:
        v17 = *(_QWORD *)(a4 + 32);
        v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v17);
        goto LABEL_39;
      }
      if ( (_BYTE)v19 != 2 )
        goto LABEL_128;
    }
LABEL_52:
    v17 = *(_QWORD *)(a4 + 32);
    v9 = (_BYTE *)(*(_QWORD *)(a4 + 24) - v17);
    goto LABEL_36;
  }
  v14 = BYTE1(v7);
  if ( !strcmp(a5, "aty") )
  {
LABEL_45:
    LOBYTE(v9) = sub_35EDAD0(v14, a4);
    return (char)v9;
  }
  if ( !strcmp(a5, "bty") )
  {
    v18 = v7 >> 16;
    goto LABEL_44;
  }
  if ( !strcmp(a5, "cty") )
  {
    v18 = v7 >> 40;
    goto LABEL_44;
  }
  v15 = strcmp(a5, "opc") == 0;
  LOBYTE(v9) = !v15;
  if ( v15 )
  {
    v16 = (unsigned __int8)v7 >> 4;
    switch ( v16 )
    {
      case 1:
        v9 = *(_BYTE **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 > 8u )
        {
          v9[8] = 99;
          *(_QWORD *)v9 = 0x706F702E646E612ELL;
          *(_QWORD *)(a4 + 32) += 9LL;
          return (char)v9;
        }
        v12 = 9;
        v13 = ".and.popc";
        goto LABEL_22;
      case 2:
        v9 = *(_BYTE **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 > 8u )
        {
          v9[8] = 99;
          *(_QWORD *)v9 = 0x706F702E726F782ELL;
          *(_QWORD *)(a4 + 32) += 9LL;
          return (char)v9;
        }
        v12 = 9;
        v13 = ".xor.popc";
        goto LABEL_22;
      case 0:
        return (char)v9;
    }
    goto LABEL_128;
  }
  if ( !strcmp(a5, "rnd") )
  {
    LOBYTE(v9) = sub_35ED6D0(*(_DWORD *)(v6 + 8) & 7, a4);
    return (char)v9;
  }
  v15 = strcmp(a5, "satf") == 0;
  LOBYTE(v9) = !v15;
  if ( v15 )
  {
    v36 = ".satfinite";
    if ( (v7 & 0x10000000) == 0 )
      return (char)v9;
    goto LABEL_91;
  }
  if ( !strcmp(a5, "scale_vec_size") )
  {
    v9 = (_BYTE *)((v7 >> 51) & 7);
    if ( (_BYTE)v9 == 1 )
    {
      v36 = ".scale_vec::2X";
    }
    else if ( (_BYTE)v9 == 2 )
    {
      v36 = ".scale_vec::4X";
    }
    else
    {
      v36 = ".scale_vec::1X";
      if ( (_BYTE)v9 )
        return (char)v9;
    }
    goto LABEL_91;
  }
  if ( !strcmp(a5, "block_scale_format") )
  {
    v36 = "ue4m3";
    if ( (BYTE6(v7) & 7) != 1 )
      v36 = "ue8m0";
    goto LABEL_91;
  }
  LODWORD(v9) = strcmp(a5, "spformat");
  if ( !(_DWORD)v9 && (v7 & 0x20000000) == 0 )
  {
    v36 = "thread";
LABEL_91:
    LOBYTE(v9) = sub_904010(a4, v36);
  }
  return (char)v9;
}
