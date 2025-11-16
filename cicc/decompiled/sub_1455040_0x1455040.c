// Function: sub_1455040
// Address: 0x1455040
//
__int64 __fastcall sub_1455040(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rbx
  int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  char v12; // dl
  bool v13; // al
  bool v14; // dl
  __int64 v15; // r14
  unsigned int v16; // r15d
  unsigned __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned int v27; // r8d
  __int64 v28; // r15
  __int64 *v29; // r14
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rax
  char v35; // dl
  bool v36; // al
  bool v37; // dl
  __int64 *v38; // rax
  __int64 v39; // rax
  char v40; // dl
  unsigned __int64 v41; // r8
  void *v42; // r9
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  bool v51; // zf
  int v52; // eax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned __int64 v57; // rdi
  void *v58; // r8
  unsigned __int64 v59; // rdi
  void *v60; // r8
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // [rsp+Ch] [rbp-44h]
  unsigned int v70; // [rsp+Ch] [rbp-44h]
  __int64 v71; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v72; // [rsp+18h] [rbp-38h]

  v5 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v5 > 0x17u )
  {
    v7 = (unsigned __int8)v5 - 24;
    switch ( (char)v5 )
    {
      case '#':
      case '%':
      case '\'':
      case ')':
      case ',':
      case '/':
      case '1':
      case '2':
      case '3':
LABEL_7:
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v8 = *(__int64 **)(a2 - 8);
        else
          v8 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v9 = *v8;
        v10 = v8[3];
        if ( (unsigned __int8)v5 <= 0x17u )
        {
          v14 = 0;
          v13 = 0;
          v41 = *(unsigned __int16 *)(a2 + 18);
          if ( (unsigned __int16)v41 <= 0x17u )
          {
            v42 = &loc_80A800;
            if ( _bittest64((const __int64 *)&v42, v41) )
              goto LABEL_12;
          }
        }
        else
        {
          if ( (unsigned __int8)v5 <= 0x2Fu )
          {
            v11 = 0x80A800000000LL;
            if ( _bittest64(&v11, v5) )
            {
LABEL_12:
              v12 = *(_BYTE *)(a2 + 17);
              v13 = (v12 & 4) != 0;
              v14 = (v12 & 2) != 0;
              goto LABEL_13;
            }
          }
          v14 = 0;
          v13 = 0;
        }
LABEL_13:
        *(_BYTE *)(a1 + 40) = 1;
        *(_DWORD *)a1 = v7;
        *(_QWORD *)(a1 + 8) = v9;
        *(_QWORD *)(a1 + 16) = v10;
        *(_BYTE *)(a1 + 24) = v13;
        *(_BYTE *)(a1 + 25) = v14;
        *(_QWORD *)(a1 + 32) = a2;
        return a1;
      case '0':
LABEL_14:
        v15 = *(_QWORD *)(sub_13CF970(a2) + 24);
        if ( *(_BYTE *)(v15 + 16) != 13 )
          goto LABEL_38;
        v69 = *(_DWORD *)(v15 + 32);
        v16 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
        if ( v69 <= 0x40 )
        {
          v17 = *(_QWORD *)(v15 + 24);
          if ( v16 > v17 )
            goto LABEL_18;
        }
        else if ( v69 - (unsigned int)sub_16A57B0(v15 + 24) <= 0x40 )
        {
          v17 = **(_QWORD **)(v15 + 24);
          if ( v16 > v17 )
          {
LABEL_18:
            v72 = v16;
            v18 = 1LL << v17;
            if ( v16 > 0x40 )
            {
              v70 = v17;
              sub_16A4EF0(&v71, 0, 0);
              if ( v72 > 0x40 )
              {
                *(_QWORD *)(v71 + 8LL * (v70 >> 6)) |= v18;
                goto LABEL_21;
              }
            }
            else
            {
              v71 = 0;
            }
            v71 |= v18;
LABEL_21:
            v19 = sub_16498A0(v15);
            v20 = sub_159C0E0(v19, &v71);
            sub_135E100(&v71);
            v21 = *(_QWORD *)sub_13CF970(a2);
            *(_BYTE *)(a1 + 40) = 1;
            *(_DWORD *)a1 = 17;
            *(_QWORD *)(a1 + 8) = v21;
            *(_QWORD *)(a1 + 16) = v20;
            *(_WORD *)(a1 + 24) = 0;
            *(_QWORD *)(a1 + 32) = 0;
            return a1;
          }
        }
LABEL_38:
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v38 = *(__int64 **)(a2 - 8);
        else
          v38 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v32 = *v38;
        v33 = v38[3];
        if ( (unsigned __int8)v5 <= 0x17u )
        {
          v37 = 0;
          v36 = 0;
          v57 = *(unsigned __int16 *)(a2 + 18);
          if ( (unsigned __int16)v57 > 0x17u )
            goto LABEL_44;
          v58 = &loc_80A800;
          if ( !_bittest64((const __int64 *)&v58, v57) )
            goto LABEL_44;
        }
        else if ( (unsigned __int8)v5 > 0x2Fu || (v39 = 0x80A800000000LL, !_bittest64(&v39, v5)) )
        {
          v37 = 0;
          v36 = 0;
          goto LABEL_44;
        }
        v40 = *(_BYTE *)(a2 + 17);
        v36 = (v40 & 4) != 0;
        v37 = (v40 & 2) != 0;
LABEL_44:
        *(_BYTE *)(a1 + 40) = 1;
        *(_DWORD *)a1 = 24;
        goto LABEL_45;
      case '4':
LABEL_27:
        v24 = sub_13CF970(a2);
        v28 = *(_QWORD *)(v24 + 24);
        v29 = (__int64 *)v24;
        if ( *(_BYTE *)(v28 + 16) == 13 && (unsigned __int8)sub_13CFF40((__int64 *)(v28 + 24), a2, v25, v26, v27) )
        {
          v30 = *v29;
          *(_BYTE *)(a1 + 40) = 1;
          *(_DWORD *)a1 = 11;
          *(_QWORD *)(a1 + 8) = v30;
          *(_QWORD *)(a1 + 16) = v28;
          *(_WORD *)(a1 + 24) = 0;
          *(_QWORD *)(a1 + 32) = 0;
          return a1;
        }
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v31 = *(__int64 **)(a2 - 8);
        else
          v31 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v32 = *v31;
        v33 = v31[3];
        if ( (unsigned __int8)v5 <= 0x17u )
        {
          v37 = 0;
          v36 = 0;
          v59 = *(unsigned __int16 *)(a2 + 18);
          if ( (unsigned __int16)v59 <= 0x17u )
          {
            v60 = &loc_80A800;
            if ( _bittest64((const __int64 *)&v60, v59) )
              goto LABEL_35;
          }
        }
        else
        {
          if ( (unsigned __int8)v5 <= 0x2Fu )
          {
            v34 = 0x80A800000000LL;
            if ( _bittest64(&v34, v5) )
            {
LABEL_35:
              v35 = *(_BYTE *)(a2 + 17);
              v36 = (v35 & 4) != 0;
              v37 = (v35 & 2) != 0;
              goto LABEL_36;
            }
          }
          v37 = 0;
          v36 = 0;
        }
LABEL_36:
        *(_BYTE *)(a1 + 40) = 1;
        *(_DWORD *)a1 = 28;
LABEL_45:
        *(_QWORD *)(a1 + 8) = v32;
        *(_QWORD *)(a1 + 16) = v33;
        *(_BYTE *)(a1 + 24) = v36;
        *(_BYTE *)(a1 + 25) = v37;
        *(_QWORD *)(a1 + 32) = a2;
        return a1;
      case 'V':
LABEL_22:
        if ( *(_DWORD *)(a2 + 64) != 1 )
          goto LABEL_3;
        if ( **(_DWORD **)(a2 + 56) )
          goto LABEL_3;
        v22 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v22 + 16) != 78 )
          goto LABEL_3;
        v23 = *(_QWORD *)(v22 - 24);
        if ( *(_BYTE *)(v23 + 16) )
          goto LABEL_3;
        switch ( *(_DWORD *)(v23 + 36) )
        {
          case 0xBD:
          case 0xD1:
            if ( (unsigned __int8)sub_14ADFA0(v22, a3) )
            {
              v65 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
              v66 = 1 - v65;
              v67 = *(_QWORD *)(v22 - 24 * v65);
              v68 = *(_QWORD *)(v22 + 24 * v66);
              if ( *(_DWORD *)(v23 + 36) == 189 )
              {
                *(_BYTE *)(a1 + 40) = 1;
                *(_DWORD *)a1 = 11;
                *(_QWORD *)(a1 + 8) = v67;
                *(_QWORD *)(a1 + 16) = v68;
                *(_WORD *)(a1 + 24) = 1;
              }
              else
              {
                *(_BYTE *)(a1 + 40) = 1;
                *(_DWORD *)a1 = 11;
                *(_QWORD *)(a1 + 8) = v67;
                *(_QWORD *)(a1 + 16) = v68;
                *(_WORD *)(a1 + 24) = 256;
              }
              *(_QWORD *)(a1 + 32) = 0;
            }
            else
            {
              v47 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
              v48 = 1 - v47;
              v49 = *(_QWORD *)(v22 - 24 * v47);
              v50 = *(_QWORD *)(v22 + 24 * v48);
              *(_BYTE *)(a1 + 40) = 1;
              *(_DWORD *)a1 = 11;
              *(_QWORD *)(a1 + 8) = v49;
              *(_QWORD *)(a1 + 16) = v50;
              *(_WORD *)(a1 + 24) = 0;
              *(_QWORD *)(a1 + 32) = 0;
            }
            break;
          case 0xC3:
          case 0xD2:
            v43 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
            v44 = 1 - v43;
            v45 = *(_QWORD *)(v22 - 24 * v43);
            v46 = *(_QWORD *)(v22 + 24 * v44);
            *(_BYTE *)(a1 + 40) = 1;
            *(_DWORD *)a1 = 15;
            *(_QWORD *)(a1 + 8) = v45;
            *(_QWORD *)(a1 + 16) = v46;
            *(_WORD *)(a1 + 24) = 0;
            *(_QWORD *)(a1 + 32) = 0;
            break;
          case 0xC6:
          case 0xD3:
            v51 = (unsigned __int8)sub_14ADFA0(v22, a3) == 0;
            v52 = *(_DWORD *)(v22 + 20);
            if ( v51 )
            {
              v53 = v52 & 0xFFFFFFF;
              v54 = 1 - v53;
              v55 = *(_QWORD *)(v22 - 24 * v53);
              v56 = *(_QWORD *)(v22 + 24 * v54);
              *(_BYTE *)(a1 + 40) = 1;
              *(_DWORD *)a1 = 13;
              *(_QWORD *)(a1 + 8) = v55;
              *(_QWORD *)(a1 + 16) = v56;
              *(_WORD *)(a1 + 24) = 0;
              *(_QWORD *)(a1 + 32) = 0;
            }
            else
            {
              v61 = v52 & 0xFFFFFFF;
              v62 = 1 - v61;
              v51 = *(_DWORD *)(v23 + 36) == 198;
              v63 = *(_QWORD *)(v22 - 24 * v61);
              v64 = *(_QWORD *)(v22 + 24 * v62);
              *(_BYTE *)(a1 + 40) = 1;
              *(_DWORD *)a1 = 13;
              *(_QWORD *)(a1 + 8) = v63;
              *(_QWORD *)(a1 + 16) = v64;
              if ( v51 )
                *(_WORD *)(a1 + 24) = 1;
              else
                *(_WORD *)(a1 + 24) = 256;
              *(_QWORD *)(a1 + 32) = 0;
            }
            break;
          default:
            goto LABEL_3;
        }
        return a1;
      default:
        goto LABEL_3;
    }
  }
  if ( (_BYTE)v5 == 5 )
  {
    v7 = *(unsigned __int16 *)(a2 + 18);
    switch ( (__int16)v7 )
    {
      case 11:
      case 13:
      case 15:
      case 17:
      case 20:
      case 23:
      case 25:
      case 26:
      case 27:
        goto LABEL_7;
      case 24:
        goto LABEL_14;
      case 28:
        goto LABEL_27;
      case 62:
        goto LABEL_22;
      default:
        break;
    }
  }
LABEL_3:
  *(_BYTE *)(a1 + 40) = 0;
  return a1;
}
