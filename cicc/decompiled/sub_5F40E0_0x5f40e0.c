// Function: sub_5F40E0
// Address: 0x5f40e0
//
__int64 *__fastcall sub_5F40E0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  char v4; // al
  __int64 v5; // rax
  _QWORD *v6; // r13
  int v7; // esi
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned __int8 v10; // dl
  __int64 j; // rax
  __int64 v12; // rdx
  __int64 *result; // rax
  char v14; // al
  _QWORD *v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // rdx
  _QWORD *v18; // rdx
  char v19; // al
  int v20; // eax
  __int64 v21; // r10
  __int64 v22; // r11
  __int64 v23; // r10
  __int64 v24; // r11
  __int64 v25; // rcx
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // rax
  __int64 v29; // r10
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  char v36; // al
  _QWORD *v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 i; // rax
  _QWORD *k; // rax
  __int64 m; // rbx
  char v45; // al
  char v46; // dl
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  char v50; // si
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // [rsp+0h] [rbp-D0h]
  __int64 v55; // [rsp+8h] [rbp-C8h]
  __int64 v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+10h] [rbp-C0h]
  __int64 v58; // [rsp+10h] [rbp-C0h]
  __int64 v59; // [rsp+10h] [rbp-C0h]
  __int64 v60; // [rsp+10h] [rbp-C0h]
  __int64 v61; // [rsp+18h] [rbp-B8h]
  _BOOL4 v62; // [rsp+18h] [rbp-B8h]
  __int64 v63; // [rsp+18h] [rbp-B8h]
  __int64 v64; // [rsp+18h] [rbp-B8h]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  __int64 v66; // [rsp+18h] [rbp-B8h]
  __int64 v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  char v70; // [rsp+37h] [rbp-99h]
  __int64 v71; // [rsp+38h] [rbp-98h]
  int v72; // [rsp+40h] [rbp-90h]
  int v73; // [rsp+44h] [rbp-8Ch]
  int v74; // [rsp+44h] [rbp-8Ch]
  __int64 v75; // [rsp+58h] [rbp-78h] BYREF
  char v76[8]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v77; // [rsp+68h] [rbp-68h]

  v75 = 0;
  v68 = a1;
  v2 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v2 + 4) == 6 )
  {
    v3 = *(_QWORD *)(v2 + 208);
    if ( v3 )
    {
      v40 = *(_QWORD *)(v2 + 24);
      v41 = v2 + 32;
      if ( !v40 )
        v40 = v41;
      v54 = *(_QWORD *)(v40 + 16);
      for ( i = v3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v67 = *(_QWORD *)(*(_QWORD *)i + 96LL);
    }
    else
    {
      v67 = 0;
      v54 = 0;
    }
  }
  else
  {
    v67 = 0;
    v3 = 0;
    v54 = 0;
  }
  v4 = *(_BYTE *)(a1 + 80);
  if ( v4 == 7 )
  {
    v6 = *(_QWORD **)(*(_QWORD *)(a1 + 88) + 120LL);
    for ( result = (__int64 *)*((unsigned __int8 *)v6 + 140);
          (_BYTE)result == 12;
          result = (__int64 *)*((unsigned __int8 *)v6 + 140) )
    {
      v6 = (_QWORD *)v6[20];
    }
    if ( (_BYTE)result != 11 )
      return result;
    if ( dword_4F077C4 != 2 || a2 )
    {
      v70 = 0;
      v36 = *((_BYTE *)v6 + 140);
      goto LABEL_110;
    }
    v70 = 0;
    v51 = v6[21];
    goto LABEL_162;
  }
  if ( v4 != 8 )
LABEL_89:
    sub_721090(a1);
  v5 = *(_QWORD *)(a1 + 88);
  v6 = *(_QWORD **)(v5 + 120);
  v70 = *(_BYTE *)(v5 + 88) & 3;
  if ( (*(_BYTE *)(v5 + 144) & 0x20) == 0 )
  {
    if ( *((_BYTE *)v6 + 140) != 12 )
      goto LABEL_7;
LABEL_143:
    v7 = 0;
    goto LABEL_112;
  }
  sub_6851C0(985, a1 + 48);
  if ( *((_BYTE *)v6 + 140) == 12 )
    goto LABEL_143;
LABEL_7:
  if ( v6[1] )
  {
    v7 = 0;
    v71 = *(_QWORD *)(*v6 + 96LL);
    v8 = *(_QWORD *)v71;
    goto LABEL_9;
  }
  if ( dword_4F077C4 != 2 || a2 )
    goto LABEL_157;
  v51 = v6[21];
  if ( *(_BYTE *)(a1 + 80) == 8 )
  {
    *(_BYTE *)(v51 + 113) = 2;
    v53 = *(_QWORD *)(a1 + 88);
    *(_QWORD *)(v51 + 120) = v53;
    *(_BYTE *)(v53 + 144) |= 0x40u;
    goto LABEL_163;
  }
LABEL_162:
  *(_BYTE *)(v51 + 113) = 1;
LABEL_163:
  if ( !unk_4D0425C )
    sub_85E680(*v6, unk_4F04C5C);
  v36 = *((_BYTE *)v6 + 140);
LABEL_110:
  if ( v36 == 12 )
  {
    v7 = 1;
LABEL_112:
    v37 = v6;
    do
      v37 = (_QWORD *)v37[20];
    while ( *((_BYTE *)v37 + 140) == 12 );
    v71 = *(_QWORD *)(*v37 + 96LL);
    v8 = *(_QWORD *)v71;
    if ( !v7 )
      goto LABEL_9;
    goto LABEL_115;
  }
LABEL_157:
  v71 = *(_QWORD *)(*v6 + 96LL);
  v8 = *(_QWORD *)v71;
LABEL_115:
  v7 = 1;
  *(_QWORD *)v71 = 0;
  v38 = dword_4D04434;
  *(_QWORD *)(v71 + 8) = 0;
  *(_QWORD *)(v71 + 24) = 0;
  *(_QWORD *)(v71 + 16) = 0;
  *(_QWORD *)(v71 + 32) = 0;
  if ( !v38 )
  {
    v39 = *(_DWORD *)(v71 + 176) & 0xFFF81DC0;
    BYTE1(v39) |= 0x60u;
    *(_DWORD *)(v71 + 176) = v39;
  }
LABEL_9:
  if ( v8 )
  {
    v73 = 0;
    v72 = 0;
    while ( 2 )
    {
      v9 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      if ( v7 )
      {
        *(_QWORD *)(v9 + 16) = 0;
        *(_QWORD *)(v9 + 24) = 0;
        sub_881D30(v9, v71 + 192);
        if ( (unsigned __int8)(*(_BYTE *)(v9 + 80) - 19) > 3u )
        {
          *(_BYTE *)(v9 + 81) &= ~0x10u;
          *(_QWORD *)(v9 + 64) = 0;
        }
      }
      a1 = v9;
      if ( (unsigned __int8)(sub_87D550(v9) - 1) > 1u || v72 )
      {
        v10 = *(_BYTE *)(v9 + 80);
      }
      else
      {
        a1 = 363;
        sub_6851C0(363, v6 + 8);
        v10 = *(_BYTE *)(v9 + 80);
        v72 = 1;
        if ( v10 > 0x18u )
          goto LABEL_89;
      }
      switch ( v10 )
      {
        case 2u:
          if ( v3 )
            sub_877E20(v9, 0, v3);
          else
            sub_877E90(v9, 0);
          *(_BYTE *)(*(_QWORD *)(v9 + 88) + 88LL) = v70 | *(_BYTE *)(*(_QWORD *)(v9 + 88) + 88LL) & 0xFC;
          goto LABEL_18;
        case 3u:
        case 4u:
        case 5u:
        case 6u:
          v61 = *(_QWORD *)(v9 + 88);
          if ( v3 )
            sub_877E20(v9, 0, v3);
          else
            sub_877E90(v9, 0);
          *(_BYTE *)(v61 + 88) = v70 | *(_BYTE *)(v61 + 88) & 0xFC;
LABEL_18:
          sub_8791D0(v9);
          sub_885FF0(v9, unk_4F04C5C, 0);
          goto LABEL_19;
        case 8u:
          v21 = *(_QWORD *)(v9 + 96);
          v22 = *(_QWORD *)(v9 + 88);
          if ( a2 && dword_4F077BC )
          {
            v60 = *(_QWORD *)(v9 + 96);
            v66 = *(_QWORD *)(v9 + 88);
            sub_5E5070(*(_QWORD *)(v22 + 120), v3, 0, 1u, (*(_BYTE *)(v22 + 145) & 0x20) != 0, v22 + 64);
            v21 = v60;
            v22 = v66;
          }
          if ( v7 )
          {
            v55 = v22;
            v57 = v21;
            v62 = dword_4F077C0 != 0;
            sub_8791D0(v9);
            sub_885FF0(v9, (unsigned int)dword_4F04C64, v62);
            v23 = v57;
            v24 = v55;
            v25 = v9;
            if ( !v3 )
              goto LABEL_98;
          }
          else
          {
            v56 = v21;
            v64 = v22;
            sub_878710(v9, v76);
            v34 = *(unsigned __int8 *)(v9 + 80);
            v77 = *(_QWORD *)(v64 + 64);
            v35 = sub_647630(v34, v76, (unsigned int)dword_4F04C64, dword_4F077C0);
            v24 = v64;
            v23 = v56;
            v25 = v35;
            *(_QWORD *)(v35 + 88) = v64;
            if ( !v3 )
            {
LABEL_98:
              v59 = v23;
              v65 = v25;
              sub_877E90(v25, 0);
              v29 = v59;
              v30 = v65;
              if ( !v59 )
                goto LABEL_99;
              goto LABEL_71;
            }
          }
          if ( !dword_4F077BC )
            goto LABEL_96;
          v26 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v24 + 40) + 32LL) + 40LL) + 32LL);
          if ( v3 == v26 )
            goto LABEL_96;
          v27 = 0;
          if ( v26 && dword_4F07588 )
          {
            v33 = *(_QWORD *)(v26 + 32);
            if ( *(_QWORD *)(v3 + 32) == v33 && v33 )
LABEL_96:
              v27 = v70 & 3;
            else
              v27 = 0;
          }
          *(_BYTE *)(v24 + 88) = v27 | *(_BYTE *)(v24 + 88) & 0xFC;
          if ( (*(_BYTE *)(v24 + 145) & 0x20) != 0 )
          {
            v28 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 600);
            *(_BYTE *)(v28 + 9) |= 0x10u;
          }
          v58 = v23;
          v63 = v25;
          sub_877E20(v25, 0, v3);
          v29 = v58;
          v30 = v63;
          if ( !v58 )
          {
LABEL_99:
            *(_QWORD *)(v30 + 96) = v68;
            goto LABEL_76;
          }
LABEL_71:
          if ( !v7 )
          {
            *(_QWORD *)(v30 + 96) = sub_5E4D30(v29, &v75, v68);
            goto LABEL_76;
          }
          while ( 1 )
          {
            if ( v68 == v29 )
              goto LABEL_76;
            if ( !*(_QWORD *)(v29 + 96) )
              break;
            v29 = *(_QWORD *)(v29 + 96);
          }
          *(_QWORD *)(v29 + 96) = v68;
LABEL_76:
          if ( (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 145LL) & 0x20) != 0 && v3 )
          {
            v31 = *(_QWORD *)(v3 + 168);
            if ( *(_BYTE *)(v3 + 140) == 11 && (*(_BYTE *)(v31 + 111) & 2) != 0 )
              sub_5E4DE0((__int64 *)v67, v9, v9 + 48);
            else
              *(_BYTE *)(v31 + 111) |= 2u;
            if ( !unk_4D0441C )
            {
              v32 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 600);
              *(_BYTE *)(v32 + 8) |= 2u;
            }
          }
LABEL_19:
          if ( !v8 )
            break;
          continue;
        case 9u:
        case 0xDu:
        case 0x15u:
          goto LABEL_19;
        case 0xAu:
        case 0x11u:
        case 0x14u:
          if ( !a2 )
          {
            sub_8791D0(v9);
            v10 = *(_BYTE *)(v9 + 80);
          }
          v20 = 0;
          if ( v10 != 17 )
            goto LABEL_50;
          v9 = *(_QWORD *)(v9 + 88);
          if ( !v9 )
            goto LABEL_19;
          v20 = 1;
LABEL_50:
          if ( v73 )
            goto LABEL_56;
          while ( 2 )
          {
            if ( *(_BYTE *)(v9 + 80) == 20 || (*(_BYTE *)(*(_QWORD *)(v9 + 88) + 193LL) & 0x10) == 0 )
            {
              v74 = v20;
              sub_6851C0(364, v6 + 8);
              v20 = v74;
              do
              {
LABEL_56:
                if ( !v20 )
                  break;
                v9 = *(_QWORD *)(v9 + 8);
              }
              while ( v9 );
              v73 = 1;
            }
            else
            {
              if ( v20 )
              {
                v9 = *(_QWORD *)(v9 + 8);
                if ( !v9 )
                  goto LABEL_19;
                continue;
              }
              v73 = 0;
            }
            goto LABEL_19;
          }
        case 0x13u:
          sub_6851C0((*(_BYTE *)(*(_QWORD *)(v9 + 88) + 265LL) & 1) == 0 ? 775 : 2687, v6 + 8);
          sub_8791D0(v9);
          goto LABEL_19;
        case 0x18u:
          sub_6851C0(3348, v6 + 8);
          sub_8791D0(v9);
          goto LABEL_19;
        default:
          goto LABEL_89;
      }
      break;
    }
  }
  for ( j = v75; j; *(_QWORD *)(v12 + 8) = 0 )
  {
    v12 = j;
    j = *(_QWORD *)(j + 8);
  }
  result = (__int64 *)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    result = &qword_4D0495C;
    if ( !qword_4D0495C )
    {
      for ( k = v6; *((_BYTE *)k + 140) == 12; k = (_QWORD *)k[20] )
        ;
      result = *(__int64 **)(k[21] + 152LL);
      if ( result )
      {
        if ( (*((_BYTE *)result + 29) & 0x20) == 0 )
        {
          for ( m = result[13]; m; m = *(_QWORD *)(m + 112) )
          {
            if ( unk_4D0426C )
            {
              if ( !*(_QWORD *)(m + 8) )
                continue;
            }
            else
            {
              result = (__int64 *)&dword_4D043B0;
              if ( dword_4D043B0 || (result = (__int64 *)&dword_4D043AC, dword_4D043AC) )
              {
                if ( !*(_QWORD *)(m + 8) )
                  continue;
              }
            }
            v45 = *(_BYTE *)(m + 140);
            if ( v45 == 12 )
            {
              result = (__int64 *)m;
              do
              {
                result = (__int64 *)result[20];
                v46 = *((_BYTE *)result + 140);
              }
              while ( v46 == 12 );
              if ( ((unsigned __int8)(v46 - 9) > 2u
                 || *((char *)result + 177) >= 0
                 || (result = (__int64 *)result[21]) == 0)
                && (*(_BYTE *)(m + 184) != 10 || (*(_BYTE *)(m + 186) & 0x40) == 0) )
              {
LABEL_173:
                v52 = 5;
                if ( unk_4D04964 )
                  v52 = unk_4F07471;
                result = (__int64 *)sub_684AA0(v52, 1055, m + 64);
              }
            }
            else
            {
              if ( (unsigned __int8)(v45 - 9) > 2u )
                goto LABEL_173;
              if ( *(char *)(m + 177) >= 0 )
                goto LABEL_173;
              result = *(__int64 **)(m + 168);
              if ( !result[21] )
                goto LABEL_173;
            }
          }
        }
      }
    }
  }
  if ( v3 )
  {
    v14 = *(_BYTE *)(v71 + 182);
    if ( (v14 & 8) != 0 )
    {
      *(_BYTE *)(v67 + 182) |= 8u;
      v14 = *(_BYTE *)(v71 + 182);
    }
    if ( (v14 & 0x10) != 0 )
    {
      *(_BYTE *)(v67 + 182) |= 0x10u;
      v14 = *(_BYTE *)(v71 + 182);
    }
    if ( (v14 & 0x20) != 0 )
    {
      *(_BYTE *)(v67 + 182) |= 0x20u;
      v14 = *(_BYTE *)(v71 + 182);
    }
    if ( (v14 & 0x40) != 0 )
    {
      *(_BYTE *)(v67 + 182) |= 0x40u;
      v14 = *(_BYTE *)(v71 + 182);
    }
    if ( v14 < 0 )
      *(_BYTE *)(v67 + 182) |= 0x80u;
    if ( (*(_BYTE *)(v71 + 183) & 1) != 0 )
      *(_BYTE *)(v67 + 183) |= 1u;
    if ( *((_BYTE *)v6 + 140) == 11 && *(_BYTE *)(v3 + 140) != 11 )
    {
      if ( v54 )
      {
        v47 = *(_QWORD *)(v54 + 16);
        if ( !v47 )
          goto LABEL_37;
      }
      else
      {
        v47 = **(_QWORD **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 24);
        if ( !v47 )
          goto LABEL_37;
      }
      v48 = 0;
      while ( 1 )
      {
        while ( *(_BYTE *)(v47 + 80) != 8 )
        {
LABEL_149:
          v47 = *(_QWORD *)(v47 + 16);
          if ( !v47 )
            goto LABEL_153;
        }
        v49 = *(_QWORD *)(v47 + 104);
        v50 = *(_BYTE *)(v49 + 28);
        *(_BYTE *)(v49 + 28) = v50 | 2;
        if ( !v48 )
        {
          v48 = v47;
          *(_BYTE *)(v49 + 28) = v50 | 6;
          goto LABEL_149;
        }
        v48 = v47;
        v47 = *(_QWORD *)(v47 + 16);
        if ( !v47 )
        {
LABEL_153:
          if ( v48 )
            *(_BYTE *)(*(_QWORD *)(v48 + 104) + 28LL) |= 8u;
          break;
        }
      }
    }
LABEL_37:
    v15 = *(_QWORD **)(v71 + 64);
    if ( v15 )
    {
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v17 = *(_QWORD **)(v16 + 272);
      if ( v17 )
        *v17 = v15;
      else
        *(_QWORD *)(v67 + 64) = v15;
      *(_QWORD *)(v71 + 64) = 0;
      do
      {
        v18 = v15;
        v15 = (_QWORD *)*v15;
      }
      while ( v15 );
      *(_QWORD *)(v16 + 272) = v18;
    }
    v19 = *(_BYTE *)(v71 + 183);
    if ( (v19 & 0x20) != 0 )
    {
      *(_BYTE *)(v71 + 183) = v19 & 0xDF;
      *(_BYTE *)(v67 + 183) |= 0x20u;
    }
    result = (__int64 *)*(unsigned int *)(v71 + 100);
    *(_DWORD *)(v67 + 100) += (_DWORD)result;
    *(_DWORD *)(v71 + 100) = 0;
  }
  return result;
}
