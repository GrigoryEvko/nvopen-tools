// Function: sub_1960590
// Address: 0x1960590
//
__int64 __fastcall sub_1960590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 *a7)
{
  unsigned __int8 v11; // al
  unsigned int v12; // edx
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  int v16; // eax
  int v17; // eax
  __int64 v18; // rax
  __int64 *v20; // rdi
  bool v21; // al
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rsi
  bool v28; // dl
  __int64 v29; // rbx
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rax
  int v34; // edx
  __int64 *v35; // rdi
  unsigned __int64 v36; // rax
  char v37; // bl
  int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rsi
  unsigned __int64 v41; // rbx
  _QWORD *v42; // rax
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // r15
  unsigned int v51; // ebx
  unsigned int v52; // r13d
  _QWORD *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 *v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rbx
  int v65; // ebx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r15
  __int64 v69; // r15
  __int64 *v70; // rbx
  __int64 v71; // rsi
  __int64 v72; // [rsp+8h] [rbp-C8h]
  __int64 v73; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v74; // [rsp+18h] [rbp-B8h]
  __int64 v75; // [rsp+20h] [rbp-B0h]
  __int64 v76; // [rsp+28h] [rbp-A8h]
  __int64 v77; // [rsp+30h] [rbp-A0h]
  int v78; // [rsp+38h] [rbp-98h]
  __int64 v79; // [rsp+38h] [rbp-98h]
  __int64 v80; // [rsp+38h] [rbp-98h]
  __int64 *v82; // [rsp+68h] [rbp-68h] BYREF
  __m128i v83; // [rsp+70h] [rbp-60h] BYREF
  __int64 v84; // [rsp+80h] [rbp-50h]
  __int64 v85; // [rsp+88h] [rbp-48h]
  __int64 v86; // [rsp+90h] [rbp-40h]

  v11 = *(_BYTE *)(a1 + 16);
  if ( v11 == 54 )
  {
    v12 = *(unsigned __int16 *)(a1 + 18);
    v82 = (__int64 *)a1;
    if ( ((v12 >> 7) & 6) == 0 && (v12 & 1) == 0 )
    {
      v13 = *(_QWORD *)(a1 - 24);
      v83.m128i_i64[1] = -1;
      v84 = 0;
      v83.m128i_i64[0] = v13;
      v85 = 0;
      v86 = 0;
      if ( (unsigned __int8)sub_134CBB0(a2, (__int64)&v83, 0) )
      {
LABEL_5:
        LODWORD(v14) = 1;
        return (unsigned int)v14;
      }
      v20 = v82;
      if ( v82[6] || *((__int16 *)v82 + 9) < 0 )
      {
        if ( sub_1625790((__int64)v82, 6) )
          goto LABEL_5;
        v20 = v82;
      }
      v21 = sub_15F32D0((__int64)v20);
      if ( a6 || !v21 )
      {
        v22 = *(v82 - 3);
        v23 = sub_15F2050((__int64)v82);
        v24 = sub_1632FA0(v23);
        v25 = v82;
        v77 = 1;
        v26 = v24;
        v27 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v27 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v43 = v77 * *(_QWORD *)(v27 + 32);
              v27 = *(_QWORD *)(v27 + 24);
              v77 = v43;
              continue;
            case 1:
              LODWORD(v29) = 16;
              break;
            case 2:
              LODWORD(v29) = 32;
              break;
            case 3:
            case 9:
              LODWORD(v29) = 64;
              break;
            case 4:
              LODWORD(v29) = 80;
              break;
            case 5:
            case 6:
              LODWORD(v29) = 128;
              break;
            case 7:
              v38 = sub_15A9520(v26, 0);
              v25 = v82;
              LODWORD(v29) = 8 * v38;
              break;
            case 0xB:
              LODWORD(v29) = *(_DWORD *)(v27 + 8) >> 8;
              break;
            case 0xD:
              v42 = (_QWORD *)sub_15A9930(v26, v27);
              v25 = v82;
              v29 = 8LL * *v42;
              break;
            case 0xE:
              v73 = *(_QWORD *)(v27 + 24);
              v76 = *(_QWORD *)(v27 + 32);
              v39 = sub_15A9FE0(v26, v73);
              v25 = v82;
              v79 = 1;
              v40 = v73;
              v41 = v39;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v40 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v49 = v79 * *(_QWORD *)(v40 + 32);
                    v40 = *(_QWORD *)(v40 + 24);
                    v79 = v49;
                    continue;
                  case 1:
                    v45 = 16;
                    goto LABEL_66;
                  case 2:
                    v45 = 32;
                    goto LABEL_66;
                  case 3:
                  case 9:
                    v45 = 64;
                    goto LABEL_66;
                  case 4:
                    v45 = 80;
                    goto LABEL_66;
                  case 5:
                  case 6:
                    v45 = 128;
                    goto LABEL_66;
                  case 7:
                    JUMPOUT(0x1960B2D);
                  case 0xB:
                    JUMPOUT(0x1960B23);
                  case 0xD:
                    v47 = (_QWORD *)sub_15A9930(v26, v40);
                    v25 = v82;
                    v45 = 8LL * *v47;
                    goto LABEL_66;
                  case 0xE:
                    v72 = *(_QWORD *)(v40 + 24);
                    v75 = *(_QWORD *)(v40 + 32);
                    v74 = (unsigned int)sub_15A9FE0(v26, v72);
                    v46 = sub_127FA20(v26, v72);
                    v25 = v82;
                    v45 = 8 * v75 * v74 * ((v74 + ((unsigned __int64)(v46 + 7) >> 3) - 1) / v74);
                    goto LABEL_66;
                  case 0xF:
                    v48 = sub_15A9520(v26, *(_DWORD *)(v40 + 8) >> 8);
                    v25 = v82;
                    v45 = (unsigned int)(8 * v48);
LABEL_66:
                    v29 = 8 * v76 * v41 * ((v41 + ((unsigned __int64)(v79 * v45 + 7) >> 3) - 1) / v41);
                    break;
                }
                break;
              }
              break;
            case 0xF:
              v44 = sub_15A9520(v26, *(_DWORD *)(v27 + 8) >> 8);
              v25 = v82;
              LODWORD(v29) = 8 * v44;
              break;
          }
          break;
        }
        v30 = *(_QWORD *)*(v25 - 3);
        if ( *(_BYTE *)(v30 + 8) == 16 )
          v30 = **(_QWORD **)(v30 + 16);
        v78 = *(_DWORD *)(v30 + 8) >> 8;
        v31 = (_QWORD *)sub_16498A0((__int64)v25);
        v32 = (__int64 *)sub_1643330(v31);
        v33 = sub_1646BA0(v32, v78);
        if ( *(_QWORD *)v22 == v33 )
        {
LABEL_79:
          v50 = *(_QWORD *)(v22 + 8);
          if ( v50 )
          {
            v80 = a3;
            v51 = v77 * v29;
            v52 = 0;
            do
            {
              ++v52;
              v53 = sub_1648700(v50);
              if ( v52 > dword_4FB0540 )
                break;
              if ( *((_BYTE *)v53 + 16) == 78 )
              {
                v54 = *(v53 - 3);
                if ( !*(_BYTE *)(v54 + 16)
                  && (*(_BYTE *)(v54 + 33) & 0x20) != 0
                  && *(_DWORD *)(v54 + 36) == 114
                  && !v53[1] )
                {
                  v55 = v53[-3 * (*((_DWORD *)v53 + 5) & 0xFFFFFFF)];
                  v56 = *(_DWORD *)(v55 + 32);
                  v57 = *(__int64 **)(v55 + 24);
                  v58 = v56 > 0x40
                      ? *v57
                      : (__int64)((_QWORD)v57 << (64 - (unsigned __int8)v56)) >> (64 - (unsigned __int8)v56);
                  if ( v51 <= 8 * (int)v58 && sub_15CC890(v80, v53[5], **(_QWORD **)(a4 + 32)) )
                    goto LABEL_5;
                }
              }
              v50 = *(_QWORD *)(v50 + 8);
            }
            while ( v50 );
          }
        }
        else
        {
          v34 = 0;
          while ( *(_BYTE *)(v22 + 16) == 71 )
          {
            if ( ++v34 > (unsigned int)dword_4FB0540 )
              break;
            v22 = *(_QWORD *)(v22 - 24);
            if ( v33 == *(_QWORD *)v22 )
              goto LABEL_79;
          }
        }
        v35 = v82;
        v36 = *(unsigned __int8 *)(*v82 + 8);
        if ( (unsigned __int8)v36 <= 0xFu )
        {
          v59 = 35454;
          if ( _bittest64(&v59, v36) )
            goto LABEL_96;
        }
        if ( (unsigned int)(v36 - 13) <= 1 || (_DWORD)v36 == 16 )
        {
          if ( sub_16435F0(*v82, 0) )
          {
LABEL_96:
            v60 = sub_15F2050(a1);
            v61 = sub_1632FA0(v60);
            v35 = v82;
            v14 = (unsigned __int64)(sub_127FA20(v61, *v82) + 7) >> 3;
LABEL_49:
            v83 = 0u;
            v84 = 0;
            sub_14A8180((__int64)v35, v83.m128i_i64, 0);
            v37 = (*(_BYTE *)(sub_135BF60(a5, *(v82 - 3), v14, &v83) + 67) >> 4) & 2;
            if ( a7 && v37 && sub_13FC1A0(a4, *(v82 - 3)) )
              sub_1960250(a7, (__int64 *)&v82);
            LOBYTE(v14) = v37 == 0;
            return (unsigned int)v14;
          }
          v35 = v82;
        }
        v14 = 0;
        goto LABEL_49;
      }
    }
LABEL_18:
    LODWORD(v14) = 0;
    return (unsigned int)v14;
  }
  v82 = 0;
  if ( v11 == 78 )
  {
    v15 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v15 + 16) && (*(_BYTE *)(v15 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v15 + 36) - 35) <= 3 )
      goto LABEL_18;
    LOBYTE(v16) = sub_15F3330(a1);
    LODWORD(v14) = v16;
    if ( (_BYTE)v16 )
      goto LABEL_18;
    v17 = sub_134CC90(a2, a1 | 4);
    if ( v17 == 4 )
      goto LABEL_5;
    if ( (v17 & 2) != 0 )
      goto LABEL_18;
    if ( (v17 & 0x30) != 0 )
    {
      v18 = *(_QWORD *)(a5 + 16);
      if ( a5 + 8 == v18 )
        goto LABEL_5;
      while ( *(_QWORD *)(v18 + 32) || (*(_BYTE *)(v18 + 67) & 0x20) == 0 )
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( a5 + 8 == v18 )
          goto LABEL_5;
      }
      goto LABEL_18;
    }
    if ( *(char *)(a1 + 23) < 0 )
    {
      v62 = sub_1648A40(a1);
      v64 = v62 + v63;
      if ( *(char *)(a1 + 23) >= 0 )
      {
        if ( (unsigned int)(v64 >> 4) )
          goto LABEL_114;
      }
      else if ( (unsigned int)((v64 - sub_1648A40(a1)) >> 4) )
      {
        if ( *(char *)(a1 + 23) < 0 )
        {
          v65 = *(_DWORD *)(sub_1648A40(a1) + 8);
          if ( *(char *)(a1 + 23) >= 0 )
            BUG();
          v66 = sub_1648A40(a1);
          v68 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v66 + v67 - 4) - v65);
LABEL_104:
          v69 = a1 + v68;
          if ( v69 != a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) )
          {
            v70 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
            while ( 1 )
            {
              v71 = *v70;
              if ( *(_BYTE *)(*(_QWORD *)*v70 + 8LL) == 15 )
              {
                v83 = 0u;
                v84 = 0;
                if ( (*(_BYTE *)(sub_135BF60(a5, v71, 0xFFFFFFFFFFFFFFFFLL, &v83) + 67) & 0x20) != 0 )
                  return (unsigned int)v14;
              }
              v70 += 3;
              if ( (__int64 *)v69 == v70 )
                goto LABEL_5;
            }
          }
          goto LABEL_5;
        }
LABEL_114:
        BUG();
      }
    }
    v68 = -24;
    goto LABEL_104;
  }
  if ( (unsigned int)v11 - 35 > 0x11 )
  {
    v28 = 1;
    if ( (unsigned __int8)(v11 - 56) <= 0x17u )
      v28 = ((0x81FFF1uLL >> (v11 - 56)) & 1) == 0;
    if ( (unsigned __int8)(v11 - 83) > 4u && (unsigned __int8)(v11 - 75) > 1u && v28 )
      goto LABEL_18;
  }
  if ( !a6 || (unsigned __int8)sub_14AF470(a1, 0, a3, 0) )
    goto LABEL_5;
  return sub_195F310(a1, a3, a4, a6, 0);
}
