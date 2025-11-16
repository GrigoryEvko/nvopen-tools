// Function: sub_6FD870
// Address: 0x6fd870
//
__int64 __fastcall sub_6FD870(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        int a5,
        int a6,
        char a7,
        char a8,
        char a9,
        char a10,
        int a11,
        _DWORD *a12,
        __int64 *a13)
{
  __int64 v16; // rbx
  char v17; // al
  char v18; // r15
  unsigned int v19; // r9d
  int v20; // r8d
  __int64 v21; // r14
  __int64 v22; // rax
  int v23; // r8d
  __int64 v24; // r13
  _DWORD *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r14
  unsigned __int16 *v30; // rsi
  __int64 v31; // rcx
  unsigned __int8 v32; // dl
  __int64 v33; // rax
  __int64 v34; // rax
  __int16 v35; // ax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v42; // rdi
  __int64 v43; // rsi
  _QWORD *v44; // r14
  __int64 v45; // rax
  int v46; // ecx
  char v47; // al
  __int64 v48; // rax
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // r12
  __int64 i; // rax
  char v53; // dl
  __int64 v54; // rdx
  char v55; // al
  __int64 v56; // rsi
  __int64 v57; // r14
  int v58; // eax
  __int64 v59; // r8
  char v60; // dl
  __int64 v61; // rax
  char v62; // dl
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rax
  _BOOL4 v69; // eax
  __int64 v70; // [rsp+8h] [rbp-1D8h]
  __int64 v71; // [rsp+8h] [rbp-1D8h]
  int v72; // [rsp+18h] [rbp-1C8h]
  __int64 v73; // [rsp+20h] [rbp-1C0h]
  unsigned int v74; // [rsp+20h] [rbp-1C0h]
  __int64 v76; // [rsp+28h] [rbp-1B8h]
  unsigned int v78; // [rsp+34h] [rbp-1ACh]
  int v79; // [rsp+38h] [rbp-1A8h]
  __int64 v80; // [rsp+48h] [rbp-198h] BYREF
  __m128i v81[25]; // [rsp+50h] [rbp-190h] BYREF

  v16 = a3;
  if ( a13 )
    *a13 = 0;
  while ( 1 )
  {
    v17 = *(_BYTE *)(a2 + 140);
    if ( v17 != 12 )
      break;
    a2 = *(_QWORD *)(a2 + 160);
  }
  if ( v17 != 7 )
  {
    if ( !a3 )
    {
      v18 = a4 & 1;
      goto LABEL_9;
    }
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0 )
    {
      v18 = a4 & 1;
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x10) == 0 || (*(_BYTE *)(a3 + 193) & 2) != 0 )
        goto LABEL_9;
      v46 = 1;
      goto LABEL_65;
    }
    *(_BYTE *)(a3 + 193) |= 0x40u;
    v46 = 1;
    goto LABEL_59;
  }
  v26 = 0;
  v27 = qword_4D03C50;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v26 = a12;
  v28 = 0;
  v73 = (__int64)v26;
  if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 8) != 0 )
  {
    v28 = 1;
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) == 0 )
    {
      v29 = *(_QWORD *)(qword_4D03C50 + 40LL);
      v30 = 0;
      sub_7ADF70(v81, 0);
      v31 = v29 - 1;
      if ( !a11 )
        goto LABEL_38;
      v32 = *(_BYTE *)(v16 + 176);
      if ( v32 > 0x26u )
      {
        if ( v32 != 43 )
          goto LABEL_38;
        v31 = v29 - 2;
        v35 = 26;
      }
      else
      {
        if ( v32 <= 0x24u )
          goto LABEL_38;
        v33 = *(_QWORD *)(a2 + 168);
        v30 = *(unsigned __int16 **)v33;
        if ( *(_BYTE *)(a2 + 140) == 12 )
        {
          v34 = a2;
          do
            v34 = *(_QWORD *)(v34 + 160);
          while ( *(_BYTE *)(v34 + 140) == 12 );
          v33 = *(_QWORD *)(v34 + 168);
        }
        if ( !*(_QWORD *)(v33 + 40) )
          v30 = *(unsigned __int16 **)v30;
        if ( !v30 )
          goto LABEL_38;
        v35 = (v32 != 37) + 31;
      }
      v30 = word_4F06418;
      if ( word_4F06418[0] != v35 )
        goto LABEL_43;
      v70 = v31;
      sub_7AE360(v81);
      sub_7B8B50(v81, word_4F06418, v36, v37);
      v31 = v70;
LABEL_38:
      if ( v31 )
      {
        v71 = v16;
        v38 = v31 - 1;
        do
        {
          if ( word_4F06418[0] != 28 )
          {
            v16 = v71;
            goto LABEL_43;
          }
          sub_7AE360(v81);
          sub_7B8B50(v81, v30, v39, v40);
        }
        while ( v38-- != 0 );
        v16 = v71;
      }
      if ( word_4F06418[0] == 28 )
      {
        sub_7BC000(v81);
        v28 = 1;
        v27 = qword_4D03C50;
        goto LABEL_44;
      }
LABEL_43:
      sub_7BC000(v81);
      v28 = 0;
      v27 = qword_4D03C50;
    }
  }
LABEL_44:
  v42 = a2;
  v43 = v73;
  if ( (unsigned int)sub_71CA50(a2, v73, 1, *(_BYTE *)(v27 + 17) & 1, v28, v16) )
  {
    if ( !dword_4D041AC )
      goto LABEL_53;
    v72 = 0;
    v44 = **(_QWORD ***)(a2 + 168);
    if ( !v44 )
      goto LABEL_53;
    do
    {
LABEL_49:
      while ( 1 )
      {
        v42 = v44[1];
        if ( (unsigned int)sub_8D5830(v42) )
        {
          v72 = 1;
          if ( (unsigned int)sub_6E5430() )
            break;
        }
        v44 = (_QWORD *)*v44;
        if ( !v44 )
          goto LABEL_52;
      }
      v43 = 603;
      v42 = 8;
      sub_5EB950(8u, 603, v44[1], v73);
      v44 = (_QWORD *)*v44;
    }
    while ( v44 );
LABEL_52:
    if ( !v72 )
    {
LABEL_53:
      if ( !v16 )
      {
        v18 = a4 & 1;
        v21 = sub_73D7F0(a2);
LABEL_70:
        if ( (unsigned int)sub_8D3D10(*a1) )
        {
          v19 = 108 - ((a6 == 0) - 1);
        }
        else
        {
          for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          v19 = 105;
          if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
            v19 = 106 - ((a6 == 0) - 1);
        }
        v20 = dword_4D041E8;
        if ( dword_4D041E8 )
        {
          v78 = v19;
          v50 = sub_6EC5C0(v21, 0);
          v19 = v78;
          v20 = v50;
          if ( v50 )
          {
            v20 = 0;
            *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x40u;
          }
        }
        goto LABEL_10;
      }
      v45 = qword_4D03C50;
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0 )
      {
        v18 = a4 & 1;
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x10) == 0 || (*(_BYTE *)(v16 + 193) & 2) != 0 )
        {
LABEL_67:
          v49 = sub_73D7F0(a2);
          v21 = v49;
          if ( (*(_BYTE *)(v16 + 207) & 0x20) != 0 )
          {
            v21 = sub_73EC50(v49);
          }
          else if ( (*(_WORD *)(v16 + 206) & 0x1010) == 0x1010 )
          {
            v21 = sub_72C930(a2);
          }
          goto LABEL_70;
        }
        v46 = 0;
        goto LABEL_65;
      }
      v46 = 0;
      v53 = *(_BYTE *)(v16 + 193) | 0x40;
      *(_BYTE *)(v16 + 193) = v53;
      if ( (v53 & 2) != 0 && dword_4F04C44 == -1 )
      {
        v54 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v54 + 6) & 6) == 0 && *(_BYTE *)(v54 + 4) != 12 && (*(_BYTE *)(v45 + 17) & 0x40) == 0 )
        {
          v55 = *(_BYTE *)(v16 + 174);
          if ( v55 == 5 )
          {
            if ( !a11 )
              goto LABEL_59;
          }
          else if ( (unsigned __int8)(v55 - 3) > 1u )
          {
            goto LABEL_59;
          }
          sub_691790(v16, 0, a12);
          v46 = 0;
        }
      }
LABEL_59:
      v47 = *(_BYTE *)(v16 + 192);
      if ( (v47 & 2) == 0 )
      {
LABEL_60:
        v18 = a4 & 1;
        goto LABEL_61;
      }
      if ( a5 )
      {
LABEL_80:
        v18 = 0;
LABEL_61:
        v48 = qword_4D03C50;
        if ( (*(_BYTE *)(v16 + 196) & 0x20) != 0 )
          *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x40u;
        if ( (*(_BYTE *)(v48 + 19) & 0x10) == 0 || (*(_BYTE *)(v16 + 193) & 2) != 0 )
        {
LABEL_66:
          if ( !v46 )
            goto LABEL_67;
LABEL_9:
          v19 = 105;
          v20 = 1;
          v21 = *(_QWORD *)&dword_4D03B80;
LABEL_10:
          v79 = v20;
          v22 = sub_73DBF0(v19, v21, a1);
          v23 = v79;
          v24 = v22;
          if ( a13 )
            *a13 = v22;
          *(_QWORD *)(v22 + 28) = *(_QWORD *)a12;
          *(_BYTE *)(v22 + 27) = (2 * (a7 & 1)) | *(_BYTE *)(v22 + 27) & 0xFD;
          *(_BYTE *)(v22 + 59) = *(_BYTE *)(v22 + 59) & 0xB0
                               | (8 * (a11 & 1))
                               | (4 * (a10 & 1))
                               | a8 & 1
                               | (2 * (a9 & 1))
                               | (v18 << 6);
          if ( unk_4D04810 )
          {
            if ( a11 && v16 && *(_BYTE *)(v16 + 174) != 4 )
            {
              sub_7377C0(*(unsigned __int8 *)(v16 + 176), &v80, v81);
              v23 = v79;
              *(_BYTE *)(v24 + 60) = *(_BYTE *)(v24 + 60) & 0xFC | v80 & 1 | (2 * (v81[0].m128i_i8[0] & 1));
            }
            else
            {
              *(_BYTE *)(v22 + 60) |= 1u;
            }
          }
          if ( !v23
            && (*(_BYTE *)(qword_4D03C50 + 20LL) & 8) == 0
            && (*(_BYTE *)(*(_QWORD *)(a2 + 168) + 16LL) & 0x20) != 0 )
          {
            v51 = sub_6ECAE0(v21, 0, 0, 1, 4u, (__int64 *)a12, &v80);
            if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
            {
              sub_6E70E0((__int64 *)v24, (__int64)v81);
              v24 = sub_6F6F40(v81, 0, v64, v65, v66, v67);
            }
            *(_QWORD *)(v80 + 56) = v24;
            return v51;
          }
          return v24;
        }
LABEL_65:
        *(_QWORD *)qword_4D03C58 = v16;
        *(_BYTE *)(qword_4D03C50 + 19LL) &= ~0x10u;
        goto LABEL_66;
      }
      if ( qword_4F04C50 )
      {
        v56 = *(_QWORD *)(qword_4F04C50 + 32LL);
        if ( (unsigned __int8)(*(_BYTE *)(v56 + 174) - 1) <= 1u )
        {
          v57 = a1[2];
          v74 = v46;
          v76 = *(_QWORD *)(*(_QWORD *)(v56 + 40) + 32LL);
          v58 = sub_8D5DF0(v76);
          v46 = v74;
          if ( v58 )
          {
            v60 = *(_BYTE *)(v57 + 24);
            v61 = v57;
            if ( v60 == 1 )
            {
              while ( 1 )
              {
                v62 = *(_BYTE *)(v61 + 56);
                if ( (unsigned __int8)(v62 - 14) > 1u && v62 != 5 )
                  break;
                v61 = *(_QWORD *)(v61 + 72);
                v60 = *(_BYTE *)(v61 + 24);
                if ( v60 != 1 )
                  goto LABEL_110;
              }
            }
            else
            {
LABEL_110:
              if ( v60 == 3 && *(_QWORD *)(v61 + 56) == *(_QWORD *)(qword_4F04C50 + 64LL) )
              {
                v68 = sub_6EC3E0(v16, v57, v76, v74, v59);
                v46 = v74;
                if ( (*(_BYTE *)(v68 + 192) & 8) != 0 )
                {
                  v69 = sub_6E53E0(5, 0x296u, a12);
                  v46 = v74;
                  if ( v69 )
                  {
                    sub_684B30(0x296u, a12);
                    v47 = *(_BYTE *)(v16 + 192);
                    v46 = v74;
                    goto LABEL_113;
                  }
                }
              }
            }
          }
          v47 = *(_BYTE *)(v16 + 192);
        }
      }
LABEL_113:
      if ( (v47 & 0x10) == 0 )
      {
        v63 = *(_QWORD *)(*(_QWORD *)(v16 + 40) + 32LL);
        if ( (unsigned __int8)(*(_BYTE *)(v63 + 140) - 9) > 2u || (*(_BYTE *)(v63 + 176) & 1) == 0 )
          goto LABEL_60;
      }
      goto LABEL_80;
    }
  }
  else if ( dword_4D041AC )
  {
    v44 = **(_QWORD ***)(a2 + 168);
    if ( v44 )
    {
      v72 = 1;
      goto LABEL_49;
    }
  }
  v24 = sub_7305B0(v42, v43);
  if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
    sub_6E50A0();
  return v24;
}
