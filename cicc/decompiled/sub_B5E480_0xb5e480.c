// Function: sub_B5E480
// Address: 0xb5e480
//
__int64 __fastcall sub_B5E480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rax
  unsigned int v8; // r15d
  int v9; // edx
  __int64 result; // rax
  __int64 v11; // rdi
  int v12; // esi
  __int64 v13; // rsi
  int v14; // esi
  int v15; // r12d
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  unsigned __int8 v20; // al
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // ecx
  __int64 v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rdi
  unsigned __int8 v30; // al
  __int64 v31; // rax
  int v32; // ebx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  _BYTE *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  int v40; // ecx
  _QWORD *v41; // rbx
  int v42; // eax
  __int64 v43; // rdi
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdi
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned int v52; // eax
  __int64 v53; // [rsp+8h] [rbp-E8h]
  __int64 v54; // [rsp+18h] [rbp-D8h]
  __int64 v55; // [rsp+2Ch] [rbp-C4h]
  __int64 v56; // [rsp+34h] [rbp-BCh]
  __int64 v57; // [rsp+3Ch] [rbp-B4h]
  __int64 v58; // [rsp+44h] [rbp-ACh]
  __int64 v59; // [rsp+4Ch] [rbp-A4h]
  __int64 v60; // [rsp+54h] [rbp-9Ch]
  __int64 v61; // [rsp+5Ch] [rbp-94h]
  _BYTE v62[12]; // [rsp+64h] [rbp-8Ch]
  _BYTE *v63; // [rsp+70h] [rbp-80h] BYREF
  __int64 v64; // [rsp+78h] [rbp-78h]
  _BYTE v65[112]; // [rsp+80h] [rbp-70h] BYREF

  v7 = *(_QWORD *)a1 + 12LL;
  v8 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  *(_QWORD *)v62 = **(_QWORD **)a1;
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  --*(_QWORD *)(a1 + 8);
  *(_DWORD *)&v62[8] = v9;
  *(_QWORD *)a1 = v7;
  switch ( *(_DWORD *)v62 )
  {
    case 0:
    case 1:
      return sub_BCB120(a3);
    case 2:
      v31 = sub_BCCE00(a3, 64);
      return sub_BCDA70(v31, 1);
    case 3:
      return sub_BCB190(a3);
    case 4:
      return sub_BCB180(a3);
    case 5:
      return sub_BCB140(a3);
    case 6:
      return sub_BCB150(a3);
    case 7:
      return sub_BCB160(a3);
    case 8:
      return sub_BCB170(a3);
    case 9:
      return sub_BCB1B0(a3, a2, *(unsigned int *)v62, *(unsigned int *)v62);
    case 0xA:
      v38 = v8;
      goto LABEL_52;
    case 0xB:
      v37 = sub_B5E480(a1, a2, a3);
      return sub_BCE1B0(v37, *(_QWORD *)&v62[4]);
    case 0xC:
      return sub_BCE3C0(a3, v8);
    case 0xD:
      v63 = v65;
      v64 = 0x800000000LL;
      if ( v8 )
      {
        v32 = 0;
        do
        {
          v33 = sub_B5E480(a1, a2, a3);
          v34 = (unsigned int)v64;
          if ( (unsigned __int64)(unsigned int)v64 + 1 > HIDWORD(v64) )
          {
            v53 = v33;
            sub_C8D5F0(&v63, v65, (unsigned int)v64 + 1LL, 8);
            v34 = (unsigned int)v64;
            v33 = v53;
          }
          ++v32;
          *(_QWORD *)&v63[8 * v34] = v33;
          v35 = (unsigned int)(v64 + 1);
          LODWORD(v64) = v64 + 1;
        }
        while ( v8 != v32 );
        v36 = v63;
      }
      else
      {
        v36 = v65;
        v35 = 0;
      }
      result = sub_BD0B90(a3, v36, v35, 0);
      if ( v63 != v65 )
      {
        v54 = result;
        _libc_free(v63, v36);
        return v54;
      }
      return result;
    case 0xE:
      return *(_QWORD *)(a2 + 8LL * (v8 >> 3));
    case 0xF:
      v39 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      v40 = *(unsigned __int8 *)(v39 + 8);
      if ( (unsigned int)(v40 - 17) <= 1 )
      {
        v41 = *(_QWORD **)(v39 + 24);
        BYTE4(v56) = (_BYTE)v40 == 18;
        LODWORD(v56) = *(_DWORD *)(v39 + 32);
        v42 = sub_BCB060(v41);
        v43 = sub_BCD140(*v41, (unsigned int)(2 * v42));
        return sub_BCE1B0(v43, v56);
      }
      v38 = (unsigned int)(2 * (*(_DWORD *)(v39 + 8) >> 8));
      goto LABEL_52;
    case 0x10:
      v28 = *(_QWORD **)(a2 + 8LL * (v8 >> 3));
      if ( (unsigned int)*((unsigned __int8 *)v28 + 8) - 17 <= 1 )
      {
        v29 = v28[3];
        v30 = *(_BYTE *)(v29 + 8);
        if ( v30 > 3u )
        {
          if ( v30 != 5 && (v30 & 0xFD) != 4 )
          {
            v50 = sub_BCAE30(v29);
            v64 = v51;
            v63 = (_BYTE *)v50;
            v52 = sub_CA1930(&v63);
            v11 = sub_BCCE00(*v28, v52 >> 1);
            goto LABEL_38;
          }
        }
        else
        {
          if ( v30 == 2 )
          {
            v11 = sub_BCB140(*v28);
            goto LABEL_38;
          }
          if ( v30 == 3 )
          {
            v11 = sub_BCB160(*v28);
LABEL_38:
            BYTE4(v57) = *((_BYTE *)v28 + 8) == 18;
            LODWORD(v57) = *((_DWORD *)v28 + 8);
            v13 = v57;
            return sub_BCE1B0(v11, v13);
          }
        }
LABEL_69:
        BUG();
      }
      v38 = *((_DWORD *)v28 + 2) >> 9;
LABEL_52:
      result = sub_BCCE00(a3, v38);
      break;
    case 0x11:
      v27 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      BYTE4(v60) = *(_BYTE *)(v27 + 8) == 18;
      LODWORD(v60) = *(_DWORD *)(v27 + 32) >> 1;
      return sub_BCE1B0(*(_QWORD *)(v27 + 24), v60);
    case 0x12:
    case 0x13:
    case 0x14:
      v11 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      v12 = *(unsigned __int8 *)(v11 + 8);
      BYTE4(v61) = (_BYTE)v12 == 18;
      LODWORD(v61) = *(_DWORD *)(v11 + 32) / (unsigned int)(2 * *(_DWORD *)v62 - 33);
      if ( (unsigned int)(v12 - 17) <= 1 )
        v11 = **(_QWORD **)(v11 + 16);
      v13 = v61;
      return sub_BCE1B0(v11, v13);
    case 0x15:
      result = sub_B5E480(a1, a2, a3);
      v25 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      v26 = *(unsigned __int8 *)(v25 + 8);
      if ( (unsigned int)(v26 - 17) <= 1 )
      {
        BYTE4(v55) = (_BYTE)v26 == 18;
        LODWORD(v55) = *(_DWORD *)(v25 + 32);
        return sub_BCE1B0(result, v55);
      }
      return result;
    case 0x16:
      return *(_QWORD *)(a2 + 8LL * HIWORD(v8));
    case 0x17:
      v24 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 > 1 )
        goto LABEL_69;
      return *(_QWORD *)(v24 + 24);
    case 0x18:
    case 0x19:
      result = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      v14 = *(unsigned __int8 *)(result + 8);
      if ( (unsigned int)(v14 - 17) > 1 )
      {
        LOBYTE(v14) = MEMORY[8];
        result = 0;
      }
      v15 = 0;
      while ( 2 )
      {
        BYTE4(v59) = (_BYTE)v14 == 18;
        LODWORD(v59) = 2 * *(_DWORD *)(result + 32);
        v17 = sub_BCE1B0(*(_QWORD *)(result + 24), v59);
        v18 = *(_QWORD *)(v17 + 24);
        v19 = (_QWORD *)v17;
        v20 = *(_BYTE *)(v18 + 8);
        if ( v20 <= 3u )
        {
          if ( v20 == 2 )
          {
            v16 = sub_BCB140(*v19);
          }
          else
          {
            if ( v20 != 3 )
              goto LABEL_69;
            v16 = sub_BCB160(*v19);
          }
        }
        else
        {
          if ( v20 == 5 || (v20 & 0xFD) == 4 )
            goto LABEL_69;
          v21 = sub_BCAE30(v18);
          v64 = v22;
          v63 = (_BYTE *)v21;
          v23 = sub_CA1930(&v63);
          v16 = sub_BCCE00(*v19, v23 >> 1);
        }
        BYTE4(v58) = *((_BYTE *)v19 + 8) == 18;
        ++v15;
        LODWORD(v58) = *((_DWORD *)v19 + 8);
        result = sub_BCE1B0(v16, v58);
        if ( (*(_DWORD *)v62 != 24) + 1 != v15 )
        {
          LOBYTE(v14) = *(_BYTE *)(result + 8);
          continue;
        }
        break;
      }
      break;
    case 0x1A:
      v44 = *(_QWORD *)(a2 + 8LL * (v8 >> 3));
      if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 > 1 )
        goto LABEL_69;
      v45 = sub_BCAE30(*(_QWORD *)(v44 + 24));
      v64 = v46;
      v63 = (_BYTE *)v45;
      v47 = sub_CA1930(&v63);
      v48 = sub_BCCE00(*(_QWORD *)v44, v47);
      v49 = *(_DWORD *)(v44 + 32);
      BYTE4(v63) = *(_BYTE *)(v44 + 8) == 18;
      LODWORD(v63) = v49;
      return sub_BCE1B0(v48, v63);
    case 0x1B:
      return sub_BCB290(a3);
    case 0x1C:
      return sub_BCB1C0(a3);
    case 0x1D:
      return sub_BCFD60(a3, (unsigned int)"aarch64.svcount", 15, 0, 0, a6, 0, 0);
    default:
      goto LABEL_69;
  }
  return result;
}
