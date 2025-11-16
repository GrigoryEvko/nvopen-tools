// Function: sub_17153B0
// Address: 0x17153b0
//
__int64 __fastcall sub_17153B0(
        __m128i *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128i a13,
        __m128 a14)
{
  __int64 v14; // rbx
  __int64 *v15; // rsi
  __int64 v16; // r12
  __int64 result; // rax
  unsigned int v18; // r13d
  int v19; // r14d
  unsigned __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // r12
  char v27; // al
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 v33; // rcx
  __int64 v34; // r8
  unsigned int v35; // r15d
  bool v36; // al
  _BYTE *v37; // r15
  unsigned int v38; // r12d
  bool v39; // al
  __int64 v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 *v43; // r15
  unsigned __int8 v44; // al
  unsigned int v45; // r12d
  bool v46; // al
  unsigned __int8 v47; // al
  unsigned int v48; // r12d
  bool v49; // al
  __int64 v50; // rax
  unsigned int v51; // r15d
  bool v52; // al
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned int v55; // r12d
  __int64 v56; // rax
  unsigned int v57; // r12d
  __int64 v58; // rax
  char v59; // cl
  unsigned int v60; // r9d
  bool v61; // al
  __int64 v62; // rax
  __int64 v63; // r8
  unsigned int v64; // r15d
  unsigned int v65; // r12d
  __int64 v66; // rax
  char v67; // cl
  unsigned int v68; // esi
  bool v69; // al
  unsigned int v70; // r15d
  __int64 v71; // rax
  char v72; // cl
  unsigned int v73; // r9d
  int v74; // eax
  bool v75; // al
  unsigned int v76; // r12d
  __int64 v77; // rax
  char v78; // cl
  unsigned int v79; // r8d
  bool v80; // al
  __int64 v81; // rax
  int v82; // eax
  __int64 v83; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v84; // [rsp-50h] [rbp-50h]
  int v85; // [rsp-4Ch] [rbp-4Ch]
  __int64 v86; // [rsp-48h] [rbp-48h] BYREF
  __int64 v87; // [rsp-40h] [rbp-40h]

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x58:
      return sub_1714690((__int64)a1, a2);
    case 0x19:
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
      {
        v14 = a2;
        v15 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v16 = *v15;
        if ( *(_BYTE *)(*v15 + 8) != 11 )
          return 0;
        sub_14C2530((__int64)&v83, v15, a1[166].m128i_i64[1], 0, a1[165].m128i_i64[0], v14, a1[166].m128i_i64[0], 0);
        v18 = v84;
        if ( v84 <= 0x40 )
        {
          v19 = sub_39FAC40(v83);
          if ( (unsigned int)v87 <= 0x40 )
          {
LABEL_7:
            if ( (unsigned int)sub_39FAC40(v86) + v19 != v18 )
              goto LABEL_8;
            goto LABEL_17;
          }
        }
        else
        {
          v19 = sub_16A5940((__int64)&v83);
          if ( (unsigned int)v87 <= 0x40 )
            goto LABEL_7;
        }
        if ( (unsigned int)sub_16A5940((__int64)&v86) + v19 != v18 )
        {
LABEL_14:
          if ( v86 )
            j_j___libc_free_0_0(v86);
LABEL_16:
          v18 = v84;
LABEL_8:
          if ( v18 > 0x40 )
          {
            if ( v83 )
              j_j___libc_free_0_0(v83);
          }
          return 0;
        }
LABEL_17:
        v20 = sub_15A3C50(v16, (__int64)&v86);
        v21 = (unsigned __int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        if ( *v21 )
        {
          v22 = v21[1];
          v23 = v21[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v23 = v22;
          if ( v22 )
            *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v23;
        }
        *v21 = v20;
        if ( v20 )
        {
          v24 = *(_QWORD *)(v20 + 8);
          v21[1] = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = (unsigned __int64)(v21 + 1) | *(_QWORD *)(v24 + 16) & 3LL;
          v21[2] = (v20 + 8) | v21[2] & 3;
          *(_QWORD *)(v20 + 8) = v21;
        }
        if ( (unsigned int)v87 <= 0x40 )
          goto LABEL_16;
        goto LABEL_14;
      }
      return 0;
    case 0x1A:
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 3 )
        return 0;
      v25 = a2;
      v26 = *(_QWORD *)(a2 - 72);
      v27 = *(_BYTE *)(v26 + 16);
      if ( v27 == 52 )
      {
        v32 = *(_QWORD *)(v26 - 48);
        v43 = *(__int64 **)(v26 - 24);
        if ( v32 )
        {
          v44 = *((_BYTE *)v43 + 16);
          if ( v44 == 13 )
          {
            v45 = *((_DWORD *)v43 + 8);
            if ( v45 <= 0x40 )
              v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45) == v43[3];
            else
              v46 = v45 == (unsigned int)sub_16A58F0((__int64)(v43 + 3));
            if ( v46 )
              goto LABEL_45;
LABEL_67:
            v47 = *(_BYTE *)(v32 + 16);
            if ( v47 == 13 )
            {
              v48 = *(_DWORD *)(v32 + 32);
              if ( v48 <= 0x40 )
                v49 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v48) == *(_QWORD *)(v32 + 24);
              else
                v49 = v48 == (unsigned int)sub_16A58F0(v32 + 24);
            }
            else
            {
              if ( *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 16 || v47 > 0x10u )
                goto LABEL_46;
              v54 = sub_15A1020((_BYTE *)v32, a2, a3, *(_QWORD *)v32);
              if ( !v54 || *(_BYTE *)(v54 + 16) != 13 )
              {
                LODWORD(v87) = *(_QWORD *)(*(_QWORD *)v32 + 32LL);
                if ( (_DWORD)v87 )
                {
                  v65 = 0;
                  do
                  {
                    v66 = sub_15A0A60(v32, v65);
                    if ( !v66 )
                      goto LABEL_46;
                    v67 = *(_BYTE *)(v66 + 16);
                    if ( v67 != 9 )
                    {
                      if ( v67 != 13 )
                        goto LABEL_46;
                      v68 = *(_DWORD *)(v66 + 32);
                      if ( v68 <= 0x40 )
                      {
                        v69 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v68) == *(_QWORD *)(v66 + 24);
                      }
                      else
                      {
                        LODWORD(v86) = *(_DWORD *)(v66 + 32);
                        v69 = (_DWORD)v86 == (unsigned int)sub_16A58F0(v66 + 24);
                      }
                      if ( !v69 )
                        goto LABEL_46;
                    }
                  }
                  while ( (_DWORD)v87 != ++v65 );
                }
LABEL_71:
                v32 = (__int64)v43;
                goto LABEL_45;
              }
              v55 = *(_DWORD *)(v54 + 32);
              if ( v55 > 0x40 )
              {
                v32 = (__int64)v43;
                if ( v55 != (unsigned int)sub_16A58F0(v54 + 24) )
                  goto LABEL_46;
                goto LABEL_45;
              }
              v49 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v55) == *(_QWORD *)(v54 + 24);
            }
            if ( !v49 )
              goto LABEL_46;
            goto LABEL_71;
          }
          if ( *(_BYTE *)(*v43 + 8) != 16 || v44 > 0x10u )
            goto LABEL_67;
          v50 = sub_15A1020(*(_BYTE **)(v26 - 24), a2, a3, *v43);
          if ( v50 && *(_BYTE *)(v50 + 16) == 13 )
          {
            v51 = *(_DWORD *)(v50 + 32);
            if ( v51 <= 0x40 )
            {
              a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v51);
              v52 = a2 == *(_QWORD *)(v50 + 24);
            }
            else
            {
              v52 = v51 == (unsigned int)sub_16A58F0(v50 + 24);
            }
            if ( v52 )
              goto LABEL_45;
          }
          else
          {
            LODWORD(v86) = *(_QWORD *)(*v43 + 32);
            if ( !(_DWORD)v86 )
              goto LABEL_45;
            LODWORD(a2) = 0;
            while ( 1 )
            {
              LODWORD(v87) = a2;
              v58 = sub_15A0A60((__int64)v43, a2);
              a2 = (unsigned int)a2;
              if ( !v58 )
                break;
              v59 = *(_BYTE *)(v58 + 16);
              if ( v59 != 9 )
              {
                if ( v59 != 13 )
                  break;
                v60 = *(_DWORD *)(v58 + 32);
                if ( v60 <= 0x40 )
                {
                  v61 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v60) == *(_QWORD *)(v58 + 24);
                }
                else
                {
                  v85 = *(_DWORD *)(v58 + 32);
                  LODWORD(v87) = a2;
                  a2 = (unsigned int)a2;
                  v61 = v85 == (unsigned int)sub_16A58F0(v58 + 24);
                }
                if ( !v61 )
                  break;
              }
              LODWORD(a2) = a2 + 1;
              if ( (_DWORD)v86 == (_DWORD)a2 )
                goto LABEL_45;
            }
          }
          v43 = *(__int64 **)(v26 - 24);
        }
        if ( !v43 )
          goto LABEL_46;
        v32 = *(_QWORD *)(v26 - 48);
        goto LABEL_67;
      }
      if ( v27 != 5 || *(_WORD *)(v26 + 18) != 28 )
        goto LABEL_31;
      v30 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
      v31 = -3 * v30;
      v32 = *(_QWORD *)(v26 - 24 * v30);
      v33 = 3 * (1 - v30);
      v34 = *(_QWORD *)(v26 + 24 * (1 - v30));
      if ( !v32 )
        goto LABEL_49;
      if ( *(_BYTE *)(v34 + 16) == 13 )
      {
        v35 = *(_DWORD *)(v34 + 32);
        if ( v35 <= 0x40 )
          v36 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v35) == *(_QWORD *)(v34 + 24);
        else
          v36 = v35 == (unsigned int)sub_16A58F0(v34 + 24);
        goto LABEL_44;
      }
      if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) == 16 )
      {
        v87 = *(_QWORD *)(v26 + 24 * (1 - v30));
        v62 = sub_15A1020((_BYTE *)v34, a2, v31, v33);
        v63 = v87;
        if ( !v62 || *(_BYTE *)(v62 + 16) != 13 )
        {
          LODWORD(v87) = *(_QWORD *)(*(_QWORD *)v87 + 32LL);
          if ( !(_DWORD)v87 )
            goto LABEL_45;
          v70 = 0;
          while ( 1 )
          {
            a2 = v70;
            v86 = v63;
            v71 = sub_15A0A60(v63, v70);
            v63 = v86;
            if ( !v71 )
              goto LABEL_48;
            v72 = *(_BYTE *)(v71 + 16);
            if ( v72 != 9 )
            {
              if ( v72 != 13 )
                goto LABEL_48;
              v73 = *(_DWORD *)(v71 + 32);
              if ( v73 <= 0x40 )
              {
                v75 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v73) == *(_QWORD *)(v71 + 24);
              }
              else
              {
                v85 = *(_DWORD *)(v71 + 32);
                v74 = sub_16A58F0(v71 + 24);
                v63 = v86;
                v75 = v85 == v74;
              }
              if ( !v75 )
                goto LABEL_48;
            }
            if ( (_DWORD)v87 == ++v70 )
              goto LABEL_45;
          }
        }
        v64 = *(_DWORD *)(v62 + 32);
        if ( v64 <= 0x40 )
        {
          a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v64);
          v36 = a2 == *(_QWORD *)(v62 + 24);
        }
        else
        {
          v36 = v64 == (unsigned int)sub_16A58F0(v62 + 24);
        }
LABEL_44:
        if ( v36 )
          goto LABEL_45;
LABEL_48:
        v30 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
LABEL_49:
        v31 = 3 * (1 - v30);
        v32 = *(_QWORD *)(v26 + 24 * (1 - v30));
        if ( !v32 )
          goto LABEL_46;
        v33 = 4 * v30;
        v37 = *(_BYTE **)(v26 - 24 * v30);
        goto LABEL_51;
      }
      v37 = *(_BYTE **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      v32 = *(_QWORD *)(v26 + 24 * (1 - v30));
LABEL_51:
      if ( v37[16] == 13 )
      {
        v38 = *((_DWORD *)v37 + 8);
        if ( v38 <= 0x40 )
          v39 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == *((_QWORD *)v37 + 3);
        else
          v39 = v38 == (unsigned int)sub_16A58F0((__int64)(v37 + 24));
        goto LABEL_54;
      }
      if ( *(_BYTE *)(*(_QWORD *)v37 + 8LL) != 16 )
        goto LABEL_46;
      v56 = sub_15A1020(v37, a2, v31, v33);
      if ( v56 && *(_BYTE *)(v56 + 16) == 13 )
      {
        v57 = *(_DWORD *)(v56 + 32);
        if ( v57 <= 0x40 )
          v39 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v57) == *(_QWORD *)(v56 + 24);
        else
          v39 = v57 == (unsigned int)sub_16A58F0(v56 + 24);
LABEL_54:
        if ( !v39 || *(_BYTE *)(v32 + 16) <= 0x10u )
          goto LABEL_46;
        goto LABEL_56;
      }
      LODWORD(v87) = *(_QWORD *)(*(_QWORD *)v37 + 32LL);
      if ( (_DWORD)v87 )
      {
        v76 = 0;
        do
        {
          v77 = sub_15A0A60((__int64)v37, v76);
          if ( !v77 )
            goto LABEL_46;
          v78 = *(_BYTE *)(v77 + 16);
          if ( v78 != 9 )
          {
            if ( v78 != 13 )
              goto LABEL_46;
            v79 = *(_DWORD *)(v77 + 32);
            if ( v79 <= 0x40 )
            {
              v80 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v79) == *(_QWORD *)(v77 + 24);
            }
            else
            {
              LODWORD(v86) = *(_DWORD *)(v77 + 32);
              v80 = (_DWORD)v86 == (unsigned int)sub_16A58F0(v77 + 24);
            }
            if ( !v80 )
              goto LABEL_46;
          }
        }
        while ( (_DWORD)v87 != ++v76 );
      }
LABEL_45:
      if ( *(_BYTE *)(v32 + 16) <= 0x10u )
      {
LABEL_46:
        if ( (*(_DWORD *)(v25 + 20) & 0xFFFFFFF) != 3 )
          return 0;
        v26 = *(_QWORD *)(v25 - 72);
LABEL_31:
        v28 = *(_BYTE *)(v26 + 16);
        if ( v28 != 13 && *(_QWORD *)(v25 - 48) == *(_QWORD *)(v25 - 24) )
        {
          v53 = sub_15A0640(*(_QWORD *)v26);
          sub_1593B40((_QWORD *)(v25 - 72), v53);
          return v25;
        }
        v29 = *(_QWORD *)(v26 + 8);
        if ( v29 && !*(_QWORD *)(v29 + 8) && (unsigned __int8)(v28 - 75) <= 1u )
        {
          switch ( *(_WORD *)(v26 + 18) & 0x7FFF )
          {
            case 3:
            case 5:
            case 6:
            case 0x21:
            case 0x23:
            case 0x25:
            case 0x27:
            case 0x29:
              *(_WORD *)(v26 + 18) = sub_15FF0F0(*(_WORD *)(v26 + 18) & 0x7FFF) | *(_WORD *)(v26 + 18) & 0x8000;
              sub_15F89F0(v25);
              sub_170B990(a1->m128i_i64[0], v26);
              result = v25;
              break;
            default:
              return 0;
          }
          return result;
        }
        return 0;
      }
LABEL_56:
      if ( *(_QWORD *)(v25 - 72) )
      {
        v40 = *(_QWORD *)(v25 - 64);
        v41 = *(_QWORD *)(v25 - 56) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v41 = v40;
        if ( v40 )
          *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
      }
      *(_QWORD *)(v25 - 72) = v32;
      v42 = *(_QWORD *)(v32 + 8);
      *(_QWORD *)(v25 - 64) = v42;
      if ( v42 )
        *(_QWORD *)(v42 + 16) = (v25 - 64) | *(_QWORD *)(v42 + 16) & 3LL;
      *(_QWORD *)(v25 - 56) = (v32 + 8) | *(_QWORD *)(v25 - 56) & 3LL;
      *(_QWORD *)(v32 + 8) = v25 - 72;
      sub_15F89F0(v25);
      return v25;
    case 0x1B:
      return sub_1708AD0(a1->m128i_i64, a2, a7, a8, a9);
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x3A:
    case 0x3B:
    case 0x49:
    case 0x4A:
    case 0x50:
    case 0x51:
    case 0x52:
      return 0;
    case 0x1D:
      return sub_1742F80(a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
    case 0x23:
      return sub_17233C0();
    case 0x24:
      return sub_17208B0();
    case 0x25:
      return sub_1725880();
    case 0x26:
      return sub_1720DB0();
    case 0x27:
      return sub_17879B0();
    case 0x28:
      return sub_1786E90();
    case 0x29:
      return sub_1784DF0();
    case 0x2A:
      return sub_1786350();
    case 0x2B:
      return sub_1785780();
    case 0x2C:
      return sub_1784060();
    case 0x2D:
      return sub_1784650();
    case 0x2E:
      return sub_1783F70();
    case 0x2F:
      return sub_17A0140();
    case 0x30:
      return sub_17A17A0();
    case 0x31:
      return sub_17A0DD0();
    case 0x32:
      return sub_17391B0();
    case 0x33:
      return sub_1735560();
    case 0x34:
      return sub_173AAB0();
    case 0x35:
      return sub_177D4A0();
    case 0x36:
      return sub_177AE90();
    case 0x37:
      return sub_1779C20();
    case 0x38:
      return sub_170F8E0((__int64)a1, a2, a7, a8, a9, a10, a11, a12, a13, a14, a3, a4, a5, a6);
    case 0x39:
      return sub_173FEF0();
    case 0x3C:
      return sub_17512B0();
    case 0x3D:
      return sub_174FAC0();
    case 0x3E:
      return sub_174EAE0();
    case 0x3F:
      return sub_174DF10();
    case 0x40:
      return sub_174DF50();
    case 0x41:
      return sub_174B7F0();
    case 0x42:
      return sub_174B800();
    case 0x43:
      return sub_1752650();
    case 0x44:
      return sub_174B7E0();
    case 0x45:
      return sub_174C6B0();
    case 0x46:
      return sub_174B810();
    case 0x47:
      return sub_1755380(a1, a2);
    case 0x48:
      return sub_174C8E0();
    case 0x4B:
      return sub_1774100();
    case 0x4C:
      return sub_1769090();
    case 0x4D:
      return sub_178F680();
    case 0x4E:
      v81 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v81 + 16) )
        return sub_1743DA0();
      v82 = *(_DWORD *)(v81 + 36);
      if ( v82 == 212 )
        return sub_1740120();
      if ( v82 == 214 )
        return sub_1740100();
      return sub_1743DA0();
    case 0x4F:
      return sub_1799C30();
    case 0x53:
      return sub_17B1E20();
    case 0x54:
      return sub_17B09A0();
    case 0x55:
      return sub_17B2B40();
    case 0x56:
      return sub_170EC90(a1, a2, a7, a8, a9, a10, a11, a12, *(double *)a13.m128i_i64, a14);
    case 0x57:
      return sub_17AF3F0();
  }
}
