// Function: sub_279CC20
// Address: 0x279cc20
//
__int64 __fastcall sub_279CC20(__int64 a1, __int64 a2)
{
  _BYTE *v4; // r13
  unsigned int v5; // r14d
  bool v6; // al
  unsigned int v7; // eax
  unsigned int v8; // r15d
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned int v18; // r13d
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 v21; // r14
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  _BYTE *v26; // r13
  char v27; // al
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rsi
  unsigned __int8 v31; // al
  _BYTE *v32; // r13
  unsigned __int8 v33; // dl
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 **v42; // r13
  __int64 *v43; // rax
  __int64 v44; // r14
  __int64 v45; // r13
  __int64 v46; // r15
  _QWORD *v47; // rax
  __int64 v48; // r14
  _QWORD *v49; // rdi
  __int64 v50; // r8
  __int64 v51; // rsi
  __int64 v52; // rax
  unsigned int v53; // ecx
  __int64 *v54; // rdx
  __int64 v55; // r10
  __int64 v56; // r15
  __int64 v57; // r13
  __int64 *v58; // rsi
  unsigned int v59; // edx
  bool v60; // al
  unsigned int v61; // esi
  bool v62; // al
  __int64 v63; // rdx
  _BYTE *v64; // rax
  unsigned int v65; // edx
  _BYTE *v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // r13
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  bool v73; // dl
  __int64 v74; // rsi
  unsigned __int8 *v75; // rax
  unsigned int v76; // edx
  int v77; // eax
  _BYTE *v78; // rax
  int v79; // eax
  int v80; // edx
  int v81; // r11d
  _BYTE *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rsi
  _BYTE *v85; // rax
  _BYTE *v86; // rdi
  char v87; // al
  int v88; // eax
  _BYTE *v89; // rax
  _BYTE *v90; // rdx
  int v91; // [rsp+0h] [rbp-70h]
  int v92; // [rsp+0h] [rbp-70h]
  __int64 v93; // [rsp+8h] [rbp-68h]
  bool v94; // [rsp+8h] [rbp-68h]
  int v95; // [rsp+8h] [rbp-68h]
  unsigned __int8 v96; // [rsp+8h] [rbp-68h]
  unsigned int v97; // [rsp+8h] [rbp-68h]
  int v98; // [rsp+10h] [rbp-60h]
  __int64 v99; // [rsp+10h] [rbp-60h]
  unsigned __int64 v100; // [rsp+10h] [rbp-60h]
  __int64 v101; // [rsp+18h] [rbp-58h]
  unsigned int v102; // [rsp+18h] [rbp-58h]
  int v103; // [rsp+18h] [rbp-58h]
  int v104; // [rsp+18h] [rbp-58h]
  __int64 v105; // [rsp+18h] [rbp-58h]
  int v106; // [rsp+18h] [rbp-58h]
  __int64 v107; // [rsp+18h] [rbp-58h]
  __int64 v108; // [rsp+18h] [rbp-58h]
  unsigned int v109; // [rsp+18h] [rbp-58h]
  unsigned __int64 v110; // [rsp+18h] [rbp-58h]
  _BYTE *v111; // [rsp+20h] [rbp-50h] BYREF
  __int64 v112; // [rsp+28h] [rbp-48h] BYREF
  __int64 v113[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v111 = v4;
  if ( *v4 != 17 )
  {
    if ( *v4 <= 0x15u )
      return 0;
    v10 = (__int64 *)sub_BD5C60((__int64)v4);
    v11 = sub_ACD6D0(v10);
    v15 = *(_QWORD *)(a2 + 40);
    v101 = v11;
    v16 = v15 + 48;
    v17 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v17 == v15 + 48 )
      goto LABEL_69;
    if ( !v17 )
      BUG();
    v93 = v17 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 > 0xA || (v98 = sub_B46E30(v17 - 24)) == 0 )
    {
LABEL_69:
      v8 = 0;
    }
    else
    {
      v18 = 0;
      v8 = 0;
      while ( 1 )
      {
        v19 = v18++;
        v20 = sub_B46EC0(v93, v19);
        v113[0] = v15;
        v113[1] = v20;
        v8 |= sub_2795CD0(a1, (__int64)v111, v101, v113, 0);
        if ( v98 == v18 )
          break;
        v15 = *(_QWORD *)(a2 + 40);
      }
    }
    v21 = a1 + 488;
    v22 = (_QWORD *)sub_279C8E0(a1 + 488, (__int64 *)&v111, v12, v16, v13, v14);
    v25 = v101;
    *v22 = v101;
    v26 = v111;
    v27 = *v111;
    if ( *v111 != 59 )
    {
LABEL_17:
      if ( (unsigned __int8)(v27 - 82) > 1u || !(unsigned __int8)sub_B52A20((__int64)v26, 0, v23, v25, v24) )
        return v8;
      v30 = *((_QWORD *)v26 - 8);
      v113[0] = v30;
      v31 = *(_BYTE *)v30;
      v32 = (_BYTE *)*((_QWORD *)v26 - 4);
      if ( *(_BYTE *)v30 <= 0x15u )
      {
        if ( *v32 <= 0x15u )
          goto LABEL_24;
        v66 = v32;
        v113[0] = (__int64)v32;
        v31 = *v32;
        v32 = (_BYTE *)v30;
        v30 = (__int64)v66;
      }
      v33 = *v32;
      if ( v31 > 0x1Cu )
      {
LABEL_21:
        if ( *v32 > 0x1Cu )
          goto LABEL_22;
        goto LABEL_101;
      }
      if ( v33 > 0x1Cu )
      {
        v113[0] = (__int64)v32;
        v31 = *v32;
        if ( *v32 != 22 )
        {
          if ( v31 <= 0x1Cu )
          {
            v90 = v32;
            v32 = (_BYTE *)v30;
            v30 = (__int64)v90;
            goto LABEL_24;
          }
          v82 = (_BYTE *)v30;
          v30 = (__int64)v32;
          v32 = v82;
          goto LABEL_21;
        }
        v89 = (_BYTE *)v30;
        v33 = *(_BYTE *)v30;
        v30 = (__int64)v32;
        v32 = v89;
      }
      else
      {
        v30 = v113[0];
        v31 = *(_BYTE *)v113[0];
        if ( *(_BYTE *)v113[0] != 22 )
          goto LABEL_24;
      }
      if ( v33 == 22 )
      {
LABEL_22:
        v34 = a1 + 136;
        v102 = sub_2792F80(v34, v30);
        if ( v102 < (unsigned int)sub_2792F80(v34, (__int64)v32) )
        {
          v83 = v113[0];
          v30 = (__int64)v32;
          v113[0] = (__int64)v32;
          v31 = *v32;
          v32 = (_BYTE *)v83;
        }
        else
        {
          v30 = v113[0];
          v31 = *(_BYTE *)v113[0];
        }
LABEL_24:
        if ( v31 <= 0x15u && *v32 <= 0x15u )
          return v8;
        goto LABEL_26;
      }
LABEL_101:
      v30 = v113[0];
LABEL_26:
      v35 = *(_QWORD *)(v30 + 16);
      v36 = *(_QWORD *)(a2 + 40);
      if ( v35 )
      {
        while ( 1 )
        {
          v37 = *(_QWORD *)(v35 + 24);
          if ( *(_BYTE *)v37 > 0x1Cu && v36 == *(_QWORD *)(v37 + 40) )
            break;
          v35 = *(_QWORD *)(v35 + 8);
          if ( !v35 )
            return v8;
        }
        *(_QWORD *)sub_279C8E0(v21, v113, v37, v36, v28, v29) = v32;
      }
      return v8;
    }
    v24 = *((_QWORD *)v111 - 8);
    if ( *(_BYTE *)v24 == 17 )
    {
      v59 = *(_DWORD *)(v24 + 32);
      if ( !v59 )
        goto LABEL_87;
      if ( v59 <= 0x40 )
      {
        v25 = 64 - v59;
        v60 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v59) == *(_QWORD *)(v24 + 24);
      }
      else
      {
        v103 = *(_DWORD *)(v24 + 32);
        v60 = v103 == (unsigned int)sub_C445E0(v24 + 24);
      }
    }
    else
    {
      v25 = *(_QWORD *)(v24 + 8);
      v63 = (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17;
      if ( (unsigned int)v63 > 1 || *(_BYTE *)v24 > 0x15u )
        goto LABEL_55;
      v99 = *(_QWORD *)(v24 + 8);
      v105 = *((_QWORD *)v111 - 8);
      v64 = sub_AD7630(v105, 0, v63);
      v24 = v105;
      v25 = v99;
      if ( !v64 || *v64 != 17 )
      {
        if ( *(_BYTE *)(v99 + 8) == 17 )
        {
          v73 = 0;
          v74 = 0;
          v91 = *(_DWORD *)(v99 + 32);
          if ( v91 )
          {
            while ( 1 )
            {
              v94 = v73;
              v107 = v24;
              v75 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)v24, v74);
              v24 = v107;
              v73 = v94;
              if ( !v75 )
                break;
              v25 = *v75;
              if ( (_BYTE)v25 != 13 )
              {
                if ( (_BYTE)v25 != 17 )
                  goto LABEL_55;
                v76 = *((_DWORD *)v75 + 8);
                if ( v76 )
                {
                  if ( v76 <= 0x40 )
                  {
                    v25 = 64 - v76;
                    v73 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v76) == *((_QWORD *)v75 + 3);
                  }
                  else
                  {
                    v95 = *((_DWORD *)v75 + 8);
                    v77 = sub_C445E0((__int64)(v75 + 24));
                    v24 = v107;
                    v73 = v95 == v77;
                  }
                  if ( !v73 )
                    goto LABEL_55;
                }
                else
                {
                  v73 = 1;
                }
              }
              v74 = (unsigned int)(v74 + 1);
              if ( v91 == (_DWORD)v74 )
              {
                if ( !v73 )
                  goto LABEL_55;
                goto LABEL_87;
              }
            }
          }
        }
        goto LABEL_55;
      }
      v65 = *((_DWORD *)v64 + 8);
      if ( !v65 )
      {
LABEL_87:
        v23 = *((_QWORD *)v26 - 4);
        if ( v23 )
        {
          v112 = *((_QWORD *)v26 - 4);
          goto LABEL_75;
        }
LABEL_56:
        if ( *(_BYTE *)v23 == 17 )
        {
          v61 = *(_DWORD *)(v23 + 32);
          if ( v61 )
          {
            if ( v61 <= 0x40 )
            {
              v25 = 64 - v61;
              v62 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v61) == *(_QWORD *)(v23 + 24);
            }
            else
            {
              v104 = *(_DWORD *)(v23 + 32);
              v62 = v104 == (unsigned int)sub_C445E0(v23 + 24);
            }
LABEL_60:
            if ( !v62 )
            {
LABEL_61:
              v26 = v111;
              v27 = *v111;
              goto LABEL_17;
            }
          }
        }
        else
        {
          v24 = *(_QWORD *)(v23 + 8);
          v108 = v24;
          v25 = (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17;
          if ( (unsigned int)v25 > 1 || *(_BYTE *)v23 > 0x15u )
            goto LABEL_61;
          v100 = v23;
          v78 = sub_AD7630(v23, 0, v23);
          v23 = v100;
          v24 = v108;
          if ( !v78 || *v78 != 17 )
          {
            if ( *(_BYTE *)(v108 + 8) == 17 )
            {
              v92 = *(_DWORD *)(v108 + 32);
              if ( v92 )
              {
                LOBYTE(v25) = 0;
                v84 = 0;
                while ( 1 )
                {
                  v96 = v25;
                  v110 = v23;
                  v85 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v23, v84);
                  v86 = v85;
                  if ( !v85 )
                    break;
                  v87 = *v85;
                  v23 = v110;
                  v25 = v96;
                  if ( v87 != 13 )
                  {
                    if ( v87 != 17 )
                      goto LABEL_61;
                    v24 = *((unsigned int *)v86 + 8);
                    if ( (_DWORD)v24 )
                    {
                      if ( (unsigned int)v24 <= 0x40 )
                      {
                        v25 = (unsigned int)(64 - v24);
                        LOBYTE(v25) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *((_QWORD *)v86 + 3);
                      }
                      else
                      {
                        v97 = *((_DWORD *)v86 + 8);
                        v88 = sub_C445E0((__int64)(v86 + 24));
                        v24 = v97;
                        v23 = v110;
                        LOBYTE(v25) = v97 == v88;
                      }
                      if ( !(_BYTE)v25 )
                        goto LABEL_61;
                    }
                    else
                    {
                      v25 = 1;
                    }
                  }
                  v84 = (unsigned int)(v84 + 1);
                  if ( v92 == (_DWORD)v84 )
                  {
                    if ( (_BYTE)v25 )
                      goto LABEL_73;
                    goto LABEL_61;
                  }
                }
              }
            }
            goto LABEL_61;
          }
          v23 = *((unsigned int *)v78 + 8);
          if ( (_DWORD)v23 )
          {
            if ( (unsigned int)v23 > 0x40 )
            {
              v109 = *((_DWORD *)v78 + 8);
              v79 = sub_C445E0((__int64)(v78 + 24));
              v23 = v109;
              v62 = v109 == v79;
              goto LABEL_60;
            }
            v25 = (unsigned int)(64 - v23);
            v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23);
            if ( *((_QWORD *)v78 + 3) != v23 )
              goto LABEL_61;
          }
        }
LABEL_73:
        if ( !*((_QWORD *)v26 - 8) )
          goto LABEL_61;
        v112 = *((_QWORD *)v26 - 8);
LABEL_75:
        v67 = (__int64 *)sub_BD5C60((__int64)v111);
        v68 = sub_ACD720(v67);
        *(_QWORD *)sub_279C8E0(a1 + 488, &v112, v69, v70, v71, v72) = v68;
        goto LABEL_61;
      }
      if ( v65 <= 0x40 )
      {
        v25 = 64 - v65;
        v60 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v65) == *((_QWORD *)v64 + 3);
      }
      else
      {
        v106 = *((_DWORD *)v64 + 8);
        v60 = v106 == (unsigned int)sub_C445E0((__int64)(v64 + 24));
      }
    }
    if ( !v60 )
    {
LABEL_55:
      v23 = *((_QWORD *)v26 - 4);
      goto LABEL_56;
    }
    goto LABEL_87;
  }
  v5 = *((_DWORD *)v4 + 8);
  if ( v5 <= 0x40 )
    v6 = *((_QWORD *)v4 + 3) == 0;
  else
    v6 = v5 == (unsigned int)sub_C444A0((__int64)(v4 + 24));
  if ( !v6 )
    goto LABEL_5;
  v41 = (_QWORD *)sub_BD5C60((__int64)v4);
  v42 = (__int64 **)sub_BCB2B0(v41);
  v43 = (__int64 *)sub_BD5C60((__int64)v111);
  v44 = sub_BCE3C0(v43, 0);
  v45 = sub_ACADE0(v42);
  v46 = sub_AD6530(v44, 0);
  v47 = sub_BD2C40(80, unk_3F10A10);
  v48 = (__int64)v47;
  if ( v47 )
    sub_B4D460((__int64)v47, v45, v46, a2 + 24, 0);
  v49 = *(_QWORD **)(a1 + 120);
  if ( !v49 )
    goto LABEL_5;
  v50 = *(_QWORD *)(a2 + 40);
  v51 = *(_QWORD *)(*v49 + 72LL);
  v52 = *(unsigned int *)(*v49 + 88LL);
  if ( (_DWORD)v52 )
  {
    v53 = (v52 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v54 = (__int64 *)(v51 + 16LL * v53);
    v55 = *v54;
    if ( v50 == *v54 )
    {
LABEL_40:
      if ( v54 != (__int64 *)(v51 + 16 * v52) )
      {
        v56 = v54[1];
        if ( v56 )
        {
          v57 = *(_QWORD *)(v56 + 8);
          if ( v57 != v56 )
          {
            while ( 1 )
            {
              if ( !v57 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v57 - 32) - 26 <= 1 && !sub_B445A0(*(_QWORD *)(v57 + 40), v48) )
                break;
              v57 = *(_QWORD *)(v57 + 8);
              if ( v56 == v57 )
              {
                v49 = *(_QWORD **)(a1 + 120);
                goto LABEL_48;
              }
            }
            v58 = (__int64 *)sub_D69520(*(_QWORD **)(a1 + 120), v48, 0, v57 - 32);
            goto LABEL_49;
          }
        }
      }
    }
    else
    {
      v80 = 1;
      while ( v55 != -4096 )
      {
        v81 = v80 + 1;
        v53 = (v52 - 1) & (v80 + v53);
        v54 = (__int64 *)(v51 + 16LL * v53);
        v55 = *v54;
        if ( v50 == *v54 )
          goto LABEL_40;
        v80 = v81;
      }
    }
  }
LABEL_48:
  v58 = (__int64 *)sub_D694D0(v49, v48, 0, *(_QWORD *)(v48 + 40), 2u, 1u);
LABEL_49:
  sub_D75120(*(__int64 **)(a1 + 120), v58, 0);
LABEL_5:
  LOBYTE(v7) = sub_CF91F0(a2);
  v8 = v7;
  if ( !(_BYTE)v7 )
    return 0;
  sub_278A7A0(a1 + 136, (_BYTE *)a2);
  v40 = *(unsigned int *)(a1 + 656);
  if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 660) )
  {
    sub_C8D5F0(a1 + 648, (const void *)(a1 + 664), v40 + 1, 8u, v38, v39);
    v40 = *(unsigned int *)(a1 + 656);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 648) + 8 * v40) = a2;
  ++*(_DWORD *)(a1 + 656);
  return v8;
}
