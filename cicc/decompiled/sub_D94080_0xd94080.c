// Function: sub_D94080
// Address: 0xd94080
//
__int64 __fastcall sub_D94080(__int64 a1, unsigned __int8 *a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v7; // rbx
  __int64 v9; // rcx
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned __int8 v19; // dl
  bool v20; // al
  bool v21; // dl
  unsigned __int8 *v22; // rsi
  __int64 v23; // r15
  unsigned int v24; // edx
  int v25; // eax
  unsigned __int64 v26; // rcx
  __int64 v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rax
  unsigned __int8 v35; // dl
  bool v36; // al
  bool v37; // dl
  unsigned __int8 *v38; // r14
  unsigned __int8 *v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdx
  char v42; // al
  char v43; // cl
  unsigned __int8 *v44; // r13
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // rax
  int v49; // eax
  int v50; // ebx
  bool v51; // r13
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned __int8 *v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdx
  bool v59; // al
  __int64 v60; // rdi
  unsigned __int64 v61; // rdx
  int v62; // esi
  unsigned __int8 *v63; // rax
  __int64 v64; // r8
  __int64 v65; // rdi
  __int64 v66; // rax
  unsigned __int8 v67; // cl
  bool v68; // al
  bool v69; // cl
  __int64 v70; // rax
  unsigned __int64 v71; // [rsp+8h] [rbp-58h]
  unsigned int v72; // [rsp+14h] [rbp-4Ch]
  unsigned int v73; // [rsp+18h] [rbp-48h]
  unsigned int v75; // [rsp+18h] [rbp-48h]
  __int64 v76; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v77; // [rsp+28h] [rbp-38h]

  v7 = *a2;
  if ( (unsigned __int8)v7 <= 0x1Cu )
  {
    if ( (_BYTE)v7 == 5 )
    {
      v9 = *((unsigned __int16 *)a2 + 1);
      switch ( (__int16)v9 )
      {
        case 13:
        case 15:
        case 17:
        case 19:
        case 22:
        case 25:
        case 27:
        case 28:
          goto LABEL_17;
        case 26:
          goto LABEL_24;
        case 29:
          goto LABEL_47;
        case 30:
          goto LABEL_39;
        case 64:
          goto LABEL_6;
        default:
          break;
      }
    }
LABEL_3:
    *(_BYTE *)(a1 + 40) = 0;
    return a1;
  }
  v9 = (unsigned int)(unsigned __int8)v7 - 29;
  switch ( (char)v7 )
  {
    case '*':
    case ',':
    case '.':
    case '0':
    case '3':
    case '6':
    case '8':
    case '9':
LABEL_17:
      if ( (a2[7] & 0x40) != 0 )
        v15 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v15 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v16 = *(_QWORD *)v15;
      v17 = *((_QWORD *)v15 + 4);
      if ( (unsigned __int8)v7 <= 0x1Cu )
      {
        v21 = 0;
        v20 = 0;
        if ( (*((_WORD *)a2 + 1) & 0xFFFD) == 0xD || (*((_WORD *)a2 + 1) & 0xFFF7) == 0x11 )
          goto LABEL_22;
      }
      else
      {
        if ( (unsigned __int8)v7 <= 0x36u )
        {
          v18 = 0x40540000000000LL;
          if ( _bittest64(&v18, v7) )
          {
LABEL_22:
            v19 = a2[1];
            v20 = (v19 & 4) != 0;
            v21 = (v19 & 2) != 0;
            goto LABEL_23;
          }
        }
        v21 = 0;
        v20 = 0;
      }
LABEL_23:
      *(_DWORD *)a1 = v9;
      *(_QWORD *)(a1 + 8) = v16;
      *(_QWORD *)(a1 + 16) = v17;
      *(_BYTE *)(a1 + 24) = v20;
      *(_BYTE *)(a1 + 25) = v21;
      *(_QWORD *)(a1 + 32) = a2;
      *(_BYTE *)(a1 + 40) = 1;
      return a1;
    case '7':
LABEL_24:
      if ( (a2[7] & 0x40) != 0 )
      {
        v22 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v23 = *((_QWORD *)v22 + 4);
        if ( *(_BYTE *)v23 != 17 )
          goto LABEL_35;
      }
      else
      {
        v32 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
        v22 = &a2[-v32];
        v23 = *(_QWORD *)&a2[-v32 + 32];
        if ( *(_BYTE *)v23 != 17 )
          goto LABEL_35;
      }
      v73 = *(_DWORD *)(v23 + 32);
      v24 = *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8;
      if ( v73 <= 0x40 )
      {
        v26 = *(_QWORD *)(v23 + 24);
        if ( v24 > v26 )
          goto LABEL_29;
      }
      else
      {
        v71 = v24;
        v72 = *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8;
        v25 = sub_C444A0(v23 + 24);
        v24 = v72;
        if ( v73 - v25 <= 0x40 )
        {
          v26 = **(_QWORD **)(v23 + 24);
          if ( v71 > v26 )
          {
LABEL_29:
            v77 = v24;
            v27 = 1LL << v26;
            if ( v24 > 0x40 )
            {
              v75 = v26;
              sub_C43690((__int64)&v76, 0, 0);
              if ( v77 > 0x40 )
              {
                *(_QWORD *)(v76 + 8LL * (v75 >> 6)) |= v27;
                goto LABEL_32;
              }
            }
            else
            {
              v76 = 0;
            }
            v76 |= v27;
LABEL_32:
            v28 = (__int64 *)sub_BD5C60(v23);
            v29 = sub_ACCFD0(v28, (__int64)&v76);
            sub_969240(&v76);
            v30 = (__int64 *)sub_986520((__int64)a2);
            *(_DWORD *)a1 = 19;
            v31 = *v30;
            goto LABEL_33;
          }
        }
      }
LABEL_35:
      v33 = *(_QWORD *)v22;
      if ( (unsigned __int8)v7 <= 0x1Cu )
      {
        v37 = 0;
        v36 = 0;
        if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 )
          goto LABEL_57;
      }
      else if ( (unsigned __int8)v7 > 0x36u || (v34 = 0x40540000000000LL, !_bittest64(&v34, v7)) )
      {
        v37 = 0;
        v36 = 0;
        goto LABEL_57;
      }
      v35 = a2[1];
      v36 = (v35 & 4) != 0;
      v37 = (v35 & 2) != 0;
LABEL_57:
      *(_DWORD *)a1 = 26;
      *(_QWORD *)(a1 + 8) = v33;
      *(_QWORD *)(a1 + 16) = v23;
      *(_BYTE *)(a1 + 24) = v36;
      *(_BYTE *)(a1 + 25) = v37;
      *(_QWORD *)(a1 + 32) = a2;
      *(_BYTE *)(a1 + 40) = 1;
      return a1;
    case ':':
LABEL_47:
      v42 = a2[7] & 0x40;
      v43 = a2[1] >> 1;
      if ( (a2[1] & 2) != 0 )
      {
        if ( v42 )
          v44 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v44 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v45 = *((_QWORD *)v44 + 4);
        v46 = *(_QWORD *)v44;
        *(_DWORD *)a1 = 13;
        *(_WORD *)(a1 + 24) = 257;
        *(_QWORD *)(a1 + 8) = v46;
        *(_QWORD *)(a1 + 16) = v45;
        *(_QWORD *)(a1 + 32) = 0;
        *(_BYTE *)(a1 + 40) = 1;
        return a1;
      }
      if ( v42 )
        v56 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v56 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v57 = *(_QWORD *)v56;
      v58 = *((_QWORD *)v56 + 4);
      v59 = 0;
      if ( (unsigned __int8)v7 <= 0x1Cu )
      {
        if ( (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 )
          goto LABEL_81;
      }
      else
      {
        if ( (unsigned __int8)v7 > 0x36u )
          goto LABEL_81;
        v60 = 0x40540000000000LL;
        if ( !_bittest64(&v60, v7) )
          goto LABEL_81;
      }
      v59 = (v43 & 2) != 0;
LABEL_81:
      *(_DWORD *)a1 = 29;
      *(_QWORD *)(a1 + 8) = v57;
      *(_QWORD *)(a1 + 16) = v58;
      *(_BYTE *)(a1 + 24) = v59;
      *(_BYTE *)(a1 + 25) = 0;
      *(_QWORD *)(a1 + 32) = a2;
      *(_BYTE *)(a1 + 40) = 1;
      return a1;
    case ';':
LABEL_39:
      if ( (a2[7] & 0x40) != 0 )
        v38 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v38 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v29 = *((_QWORD *)v38 + 4);
      if ( *(_BYTE *)v29 != 17 || !(unsigned __int8)sub_986B30((__int64 *)(v29 + 24), (__int64)a2, (__int64)a3, v9, a5) )
      {
        if ( sub_BCAC40(*((_QWORD *)a2 + 1), 1) )
        {
          if ( (a2[7] & 0x40) != 0 )
            v39 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v39 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v40 = *((_QWORD *)v39 + 4);
          v41 = *(_QWORD *)v39;
          *(_DWORD *)a1 = 13;
          *(_WORD *)(a1 + 24) = 0;
          *(_QWORD *)(a1 + 8) = v41;
          *(_QWORD *)(a1 + 16) = v40;
          *(_QWORD *)(a1 + 32) = 0;
          *(_BYTE *)(a1 + 40) = 1;
          return a1;
        }
        v61 = *a2;
        if ( (unsigned __int8)v61 <= 0x1Cu )
          v62 = *((unsigned __int16 *)a2 + 1);
        else
          v62 = (unsigned __int8)v61 - 29;
        if ( (a2[7] & 0x40) != 0 )
          v63 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v63 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        v64 = *(_QWORD *)v63;
        v65 = *((_QWORD *)v63 + 4);
        if ( (unsigned __int8)v61 <= 0x1Cu )
        {
          v69 = 0;
          v68 = 0;
          if ( (_BYTE)v61 != 5 || (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD && (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 )
            goto LABEL_90;
        }
        else if ( (unsigned __int8)v61 > 0x36u || (v66 = 0x40540000000000LL, !_bittest64(&v66, v61)) )
        {
          v69 = 0;
          v68 = 0;
          goto LABEL_90;
        }
        v67 = a2[1];
        v68 = (v67 & 4) != 0;
        v69 = (v67 & 2) != 0;
LABEL_90:
        *(_DWORD *)a1 = v62;
        *(_QWORD *)(a1 + 8) = v64;
        *(_QWORD *)(a1 + 16) = v65;
        *(_BYTE *)(a1 + 24) = v68;
        *(_BYTE *)(a1 + 25) = v69;
        *(_QWORD *)(a1 + 32) = a2;
        *(_BYTE *)(a1 + 40) = 1;
        return a1;
      }
      *(_DWORD *)a1 = 13;
      v31 = *(_QWORD *)v38;
LABEL_33:
      *(_QWORD *)(a1 + 8) = v31;
      *(_QWORD *)(a1 + 16) = v29;
      *(_WORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_BYTE *)(a1 + 40) = 1;
      return a1;
    case ']':
LABEL_6:
      if ( *((_DWORD *)a2 + 20) == 1
        && !**((_DWORD **)a2 + 9)
        && (v47 = *((_QWORD *)a2 - 4), *(_BYTE *)v47 == 85)
        && (v48 = *(_QWORD *)(v47 - 32)) != 0
        && !*(_BYTE *)v48
        && *(_QWORD *)(v48 + 24) == *(_QWORD *)(v47 + 80)
        && (*(_BYTE *)(v48 + 33) & 0x20) != 0 )
      {
        v49 = *(_DWORD *)(v48 + 36);
        if ( v49 != 312 )
        {
          switch ( v49 )
          {
            case 333:
            case 339:
            case 360:
            case 369:
            case 372:
              break;
            default:
              goto LABEL_7;
          }
        }
        v50 = sub_B5B5E0(*((_QWORD *)a2 - 4));
        v51 = sub_B5B640(v47);
        if ( v50 != 17 && (unsigned __int8)sub_98C6D0(v47, a3) )
        {
          v52 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
          v53 = 1 - v52;
          v54 = *(_QWORD *)(v47 - 32 * v52);
          v55 = *(_QWORD *)(v47 + 32 * v53);
          *(_DWORD *)a1 = v50;
          *(_BYTE *)(a1 + 24) = v51;
          *(_QWORD *)(a1 + 8) = v54;
          *(_QWORD *)(a1 + 16) = v55;
          *(_BYTE *)(a1 + 25) = !v51;
          *(_QWORD *)(a1 + 32) = 0;
          *(_BYTE *)(a1 + 40) = 1;
          return a1;
        }
        v70 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
        v13 = *(_QWORD *)(v47 + 32 * (1 - v70));
        v14 = *(_QWORD *)(v47 - 32 * v70);
        *(_DWORD *)a1 = v50;
      }
      else
      {
LABEL_7:
        if ( (unsigned __int8)v7 <= 0x1Cu )
          goto LABEL_3;
LABEL_8:
        if ( (_BYTE)v7 != 85 )
          goto LABEL_3;
        v10 = *((_QWORD *)a2 - 4);
        if ( !v10
          || *(_BYTE *)v10
          || *(_QWORD *)(v10 + 24) != *((_QWORD *)a2 + 10)
          || (*(_BYTE *)(v10 + 33) & 0x20) == 0
          || *(_DWORD *)(v10 + 36) != 222 )
        {
          goto LABEL_3;
        }
        v11 = *((_DWORD *)a2 + 1);
        *(_DWORD *)a1 = 15;
        v12 = v11 & 0x7FFFFFF;
        v13 = *(_QWORD *)&a2[32 * (1 - v12)];
        v14 = *(_QWORD *)&a2[-32 * v12];
      }
      *(_QWORD *)(a1 + 8) = v14;
      *(_QWORD *)(a1 + 16) = v13;
      *(_WORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_BYTE *)(a1 + 40) = 1;
      return a1;
    default:
      goto LABEL_8;
  }
}
