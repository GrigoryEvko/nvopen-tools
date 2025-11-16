// Function: sub_717910
// Address: 0x717910
//
__int64 __fastcall sub_717910(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4, _DWORD *a5, __int64 a6)
{
  _QWORD *v7; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  char v11; // al
  unsigned int v12; // r14d
  int v13; // eax
  __int64 v15; // rdi
  __int64 i; // r15
  unsigned __int16 v17; // cx
  __int64 v18; // r8
  __int64 v19; // r14
  __int64 v20; // rdi
  __m128i *v21; // rax
  __int64 v22; // r14
  bool v23; // al
  unsigned __int16 v24; // cx
  __int16 v25; // bx
  __int64 v26; // rdx
  int v27; // ecx
  __int64 v28; // r14
  unsigned __int8 v29; // bl
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rsi
  bool v38; // al
  const __m128i *v39; // r14
  __int64 mm; // rax
  __int64 nn; // r9
  const __m128i *v42; // r14
  signed __int64 v43; // rax
  __int64 n; // r14
  __int64 v45; // r15
  unsigned __int64 v46; // rdx
  __int64 ii; // rax
  __int64 v48; // r8
  unsigned __int16 v49; // ax
  __int64 v50; // rsi
  bool v51; // al
  __m128i *v52; // rbx
  __m128i *v53; // rax
  __m128i *v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // rcx
  unsigned __int64 v57; // rsi
  __int64 v58; // rax
  char k; // dl
  _QWORD *v60; // rbx
  __int64 v61; // rdx
  char m; // al
  __int64 v63; // rdi
  const __m128i *v64; // rdx
  _QWORD *v65; // rax
  __int64 jj; // r14
  const __m128i *v67; // rdx
  __int64 kk; // rax
  unsigned __int16 v69; // cx
  unsigned __int8 v70; // al
  __int64 v71; // rax
  int v72; // eax
  _QWORD *v73; // r8
  int v74; // eax
  int v75; // eax
  _QWORD *v76; // r8
  const __m128i *v77; // r15
  __int64 v78; // rax
  unsigned int v79; // r14d
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  const __m128i *v84; // rbx
  unsigned __int8 v85; // si
  unsigned int *v86; // rbx
  __int64 j; // rax
  __int64 v88; // [rsp+10h] [rbp-B0h]
  __int64 v89; // [rsp+18h] [rbp-A8h]
  __int64 v90; // [rsp+20h] [rbp-A0h]
  __int64 v91; // [rsp+20h] [rbp-A0h]
  const __m128i *v92; // [rsp+28h] [rbp-98h]
  bool v93; // [rsp+30h] [rbp-90h]
  __int64 v94; // [rsp+30h] [rbp-90h]
  _QWORD *v95; // [rsp+30h] [rbp-90h]
  _QWORD *v96; // [rsp+30h] [rbp-90h]
  int v97; // [rsp+30h] [rbp-90h]
  unsigned int v98; // [rsp+30h] [rbp-90h]
  int v100; // [rsp+48h] [rbp-78h] BYREF
  int v101; // [rsp+4Ch] [rbp-74h] BYREF
  __int64 v102; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v103; // [rsp+58h] [rbp-68h] BYREF
  __m128i v104; // [rsp+60h] [rbp-60h] BYREF
  __m128i v105; // [rsp+70h] [rbp-50h] BYREF
  __m128i v106[4]; // [rsp+80h] [rbp-40h] BYREF

  v7 = a2;
  v9 = a1;
  v10 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  *a5 = 0;
  v102 = v10;
  if ( (*(_BYTE *)(a1 + 202) & 4) != 0 )
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8LL);
  if ( *(_BYTE *)(v9 + 174) || !*(_WORD *)(v9 + 176) )
    goto LABEL_4;
  v15 = *(_QWORD *)(v9 + 152);
  for ( i = sub_73D790(v15); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v17 = *(_WORD *)(v9 + 176);
  if ( a2 )
  {
    v18 = a2[2];
    if ( v17 <= 0x11AFu )
    {
      if ( v17 <= 0x11ADu )
      {
        if ( v17 > 0x10C0u )
        {
          switch ( v17 )
          {
            case 0x112Bu:
            case 0x112Cu:
            case 0x1133u:
              v51 = v18 == 0;
              goto LABEL_181;
            case 0x112Du:
            case 0x112Eu:
            case 0x112Fu:
            case 0x1130u:
            case 0x1131u:
            case 0x1132u:
            case 0x1134u:
            case 0x1135u:
            case 0x1136u:
            case 0x1137u:
            case 0x1138u:
            case 0x1139u:
            case 0x113Au:
            case 0x113Bu:
            case 0x113Cu:
            case 0x113Du:
            case 0x113Eu:
            case 0x113Fu:
            case 0x1140u:
            case 0x1141u:
            case 0x1142u:
            case 0x1143u:
            case 0x1144u:
            case 0x1145u:
            case 0x1146u:
            case 0x1147u:
            case 0x1148u:
            case 0x1149u:
            case 0x114Au:
            case 0x114Bu:
            case 0x114Cu:
            case 0x114Du:
            case 0x114Eu:
            case 0x114Fu:
            case 0x1151u:
            case 0x1152u:
            case 0x1153u:
            case 0x1154u:
            case 0x1155u:
            case 0x1156u:
            case 0x1157u:
            case 0x1158u:
            case 0x1159u:
            case 0x115Au:
            case 0x115Bu:
            case 0x115Cu:
            case 0x115Du:
            case 0x115Eu:
            case 0x115Fu:
            case 0x1160u:
            case 0x1161u:
            case 0x1162u:
            case 0x1164u:
            case 0x1165u:
            case 0x1168u:
            case 0x1169u:
            case 0x116Au:
            case 0x116Bu:
            case 0x116Cu:
            case 0x116Eu:
            case 0x1171u:
            case 0x1172u:
            case 0x1173u:
            case 0x1174u:
            case 0x1175u:
            case 0x1176u:
              goto LABEL_4;
            case 0x1150u:
              goto LABEL_175;
            case 0x1163u:
            case 0x1166u:
            case 0x1167u:
              goto LABEL_73;
            case 0x116Du:
              goto LABEL_195;
            case 0x116Fu:
            case 0x1170u:
            case 0x1177u:
              if ( !v18 || *(_QWORD *)(v18 + 16) )
                goto LABEL_179;
              v95 = (_QWORD *)a2[2];
              v72 = sub_8D2AC0(*a2);
              v73 = v95;
              if ( !v72 )
              {
                v74 = sub_8D3D40(*a2);
                v73 = v95;
                if ( !v74 )
                  goto LABEL_121;
              }
              v96 = v73;
              v75 = sub_8D2AC0(*v73);
              v76 = v96;
              if ( v75 )
                goto LABEL_255;
              if ( !(unsigned int)sub_8D3D40(*v96) )
                goto LABEL_121;
              v76 = v96;
LABEL_255:
              v11 = *((_BYTE *)a2 + 24);
              if ( v11 != 2 )
                goto LABEL_6;
              v77 = (const __m128i *)a2[7];
              if ( v77[10].m128i_i8[13] != 3 || *((_BYTE *)v76 + 24) != 2 || *(_BYTE *)(v76[7] + 173LL) != 3 )
                goto LABEL_6;
              v91 = v76[7];
              v78 = sub_8D21C0(v77[8].m128i_i64[0]);
              v79 = *(unsigned __int8 *)(v78 + 160);
              v97 = sub_70C5B0(*(unsigned __int8 *)(v78 + 160), (unsigned int *)&v77[11]);
              if ( v97 == (unsigned int)sub_70C5B0(v79, (unsigned int *)(v91 + 176)) )
              {
                v12 = 1;
                *(__m128i *)(v102 + 176) = _mm_loadu_si128(v77 + 11);
                goto LABEL_31;
              }
              sub_70BAF0(v79, v77 + 11, (_OWORD *)(v102 + 176), &v105, v106);
              if ( !(v106[0].m128i_i32[0] | v105.m128i_i32[0]) )
                goto LABEL_63;
              goto LABEL_5;
            default:
              goto LABEL_5;
          }
        }
        if ( v17 > 0x10BDu )
        {
          if ( v18 )
            goto LABEL_5;
          v11 = *((_BYTE *)a2 + 24);
          if ( dword_4F06BA0 == 8 && v11 == 2 )
          {
            v64 = (const __m128i *)a2[7];
            if ( v64[10].m128i_i8[13] == 1 )
            {
              v12 = sub_622BA0(dword_3C13D80[(unsigned __int16)(v17 - 4286)], v64 + 11, (_WORD *)(v102 + 176));
              if ( v12 )
                goto LABEL_31;
              goto LABEL_5;
            }
          }
        }
        else if ( v17 == 4139 )
        {
          if ( !v18 )
            goto LABEL_5;
          v65 = *(_QWORD **)(v18 + 16);
          if ( !v65 )
            goto LABEL_5;
          if ( v65[2] )
          {
            *a5 = 140;
            v11 = *((_BYTE *)a2 + 24);
          }
          else
          {
            v94 = a2[2];
            if ( (unsigned int)sub_8D2780(*v65) || (unsigned int)sub_8DBE70(**(_QWORD **)(v94 + 16)) )
              goto LABEL_5;
            *a5 = 2394;
            v11 = *((_BYTE *)a2 + 24);
          }
        }
        else
        {
          if ( v17 > 0x102Bu )
          {
            if ( v17 != 4174 )
            {
              if ( v17 != 4235 )
                goto LABEL_5;
              goto LABEL_26;
            }
            if ( *((_BYTE *)a2 + 24) == 2 )
            {
              v63 = a2[7];
              v19 = v102;
              if ( *(_BYTE *)(v63 + 173) == 1 )
              {
                v71 = sub_620FD0(v63, v106);
                if ( ((v71 - 4) & 0xFFFFFFFFFFFFFFFBLL) != 0 && (unsigned __int64)(v71 - 1) > 1 )
                {
                  sub_72BBE0(v102, 0, *(unsigned __int8 *)(i + 160));
                  goto LABEL_63;
                }
                goto LABEL_200;
              }
              v11 = *((_BYTE *)a2 + 24);
              goto LABEL_6;
            }
            goto LABEL_38;
          }
          if ( v17 != 3387 )
            goto LABEL_5;
          v11 = *((_BYTE *)a2 + 24);
          if ( !v18 && v11 == 2 )
          {
            v22 = a2[7];
            if ( *(_BYTE *)(v22 + 173) == 1 )
            {
              if ( !(unsigned int)sub_8D2780(*(_QWORD *)(v22 + 128)) )
                goto LABEL_5;
              if ( !(unsigned int)sub_8D2780(i) )
                goto LABEL_5;
              v106[0].m128i_i32[0] = 0;
              sub_72A510(v22, v102);
              *(_QWORD *)(v102 + 128) = i;
              if ( (int)sub_6210B0(v22, 0) < 0 )
              {
                sub_621710((__int16 *)(v102 + 176), (_BOOL4 *)v106[0].m128i_i32);
                if ( v106[0].m128i_i32[0] )
                  goto LABEL_5;
                for ( j = *(_QWORD *)(v102 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                  ;
                if ( !sub_621140(v102, v102, *(_BYTE *)(j + 160)) )
                  goto LABEL_5;
              }
              goto LABEL_62;
            }
          }
        }
LABEL_6:
        v12 = 0;
        while ( v11 == 2 )
        {
          v13 = sub_731EE0(v7);
          v7 = (_QWORD *)v7[2];
          if ( v13 )
            v12 = 1;
          if ( !v7 )
          {
            if ( !v12 )
              goto LABEL_12;
            sub_70FD90(a3, v102);
            goto LABEL_31;
          }
          v11 = *((_BYTE *)v7 + 24);
        }
        goto LABEL_12;
      }
      if ( sub_7175E0((__int64)a2, 0) )
        goto LABEL_5;
      goto LABEL_125;
    }
    if ( v17 <= 0x27F6u )
    {
      if ( v17 > 0x27C5u )
      {
        switch ( v17 )
        {
          case 0x27C6u:
          case 0x27C7u:
          case 0x27CEu:
            goto LABEL_5;
          case 0x27E6u:
          case 0x27EBu:
          case 0x27EDu:
          case 0x27EEu:
          case 0x27F3u:
          case 0x27F4u:
          case 0x27F5u:
          case 0x27F6u:
            goto LABEL_117;
          default:
            goto LABEL_4;
        }
      }
      if ( v17 == 4802 )
      {
        v56 = a2;
        v57 = 0;
        while ( 1 )
        {
          v58 = *v56;
          for ( k = *(_BYTE *)(*v56 + 140LL); k == 12; k = *(_BYTE *)(v58 + 140) )
            v58 = *(_QWORD *)(v58 + 160);
          if ( !k )
            goto LABEL_5;
          v56 = (_QWORD *)v56[2];
          ++v57;
          if ( !v56 )
          {
            if ( v57 > 4 )
            {
              if ( v57 == 5 || v57 > 6 )
              {
                *a5 = 1879;
              }
              else
              {
                v60 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 16) + 16LL) + 16LL) + 16LL);
                if ( !(unsigned int)sub_8D2AC0(*v60) && !(unsigned int)sub_8D3D40(*v60) )
                {
                  v61 = *v60;
                  for ( m = *(_BYTE *)(*v60 + 140LL); m == 12; m = *(_BYTE *)(v61 + 140) )
                    v61 = *(_QWORD *)(v61 + 160);
                  if ( m )
                    *a5 = 1880;
                }
              }
            }
            goto LABEL_5;
          }
        }
      }
      if ( v17 > 0x12C2u )
      {
LABEL_5:
        v11 = *((_BYTE *)v7 + 24);
        goto LABEL_6;
      }
      if ( v17 != 4715 )
      {
        if ( v17 > 0x126Bu )
        {
          if ( v17 != 4737 && (unsigned __int16)(v17 - 4740) > 1u )
            goto LABEL_5;
          goto LABEL_73;
        }
        if ( v17 > 0x1264u )
          goto LABEL_5;
        if ( v17 <= 0x1262u )
        {
          if ( v17 != 4586 && (unsigned __int16)(v17 - 4589) > 1u )
            goto LABEL_5;
LABEL_73:
          v23 = v18 == 0;
LABEL_155:
          if ( !a2 )
            goto LABEL_12;
          if ( !v23 )
            goto LABEL_4;
          for ( n = sub_73D790(*(_QWORD *)(v9 + 152)); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
            ;
          v11 = *((_BYTE *)a2 + 24);
          if ( v11 != 2 )
            goto LABEL_6;
          v45 = a2[7];
          if ( *(_BYTE *)(v45 + 173) != 1 )
            goto LABEL_6;
          v46 = sub_620FD0(a2[7], &v105);
          if ( v105.m128i_i32[0] )
            goto LABEL_5;
          for ( ii = *(_QWORD *)(v45 + 128); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
            ;
          v48 = *(_QWORD *)(ii + 128) * dword_4F06BA0;
          if ( !v48 )
          {
            v50 = 0;
LABEL_276:
            v81 = *(unsigned __int8 *)(n + 160);
            v12 = 1;
            sub_72BBE0(v102, v50, v81);
            goto LABEL_31;
          }
          v49 = *(_WORD *)(v9 + 176);
          v50 = 0;
          v15 = 0;
          while ( 1 )
          {
            if ( v49 > 0x1285u )
            {
              if ( v49 == 15598 )
                goto LABEL_269;
              if ( v49 > 0x3CEEu )
              {
                if ( (unsigned __int16)(v49 - 15601) > 1u )
                  goto LABEL_175;
LABEL_269:
                v50 -= ((v46 & 1) == 0) - 1LL;
                goto LABEL_168;
              }
              if ( v49 != 15593 && (unsigned __int16)(v49 - 15596) > 1u )
                goto LABEL_175;
              if ( (v46 & 1) != 0 )
                v50 = ((_BYTE)v50 + 1) & 1;
            }
            else
            {
              if ( v49 <= 0x1283u )
              {
                if ( v49 == 4586 )
                  goto LABEL_166;
                if ( v49 <= 0x11EAu )
                {
                  if ( v49 != 4451 && (unsigned __int16)(v49 - 4454) > 1u )
                    goto LABEL_175;
                  if ( (v46 & 1) != 0 )
                  {
                    v50 = 0;
                    goto LABEL_168;
                  }
                  goto LABEL_167;
                }
                if ( v49 <= 0x11EEu )
                {
                  if ( v49 <= 0x11ECu )
                    goto LABEL_175;
LABEL_166:
                  if ( (v46 & 1) != 0 )
                    goto LABEL_276;
LABEL_167:
                  ++v50;
                  goto LABEL_168;
                }
                if ( v49 != 4737 )
                  goto LABEL_175;
              }
              if ( (v46 & 1) != 0 )
              {
                v50 = v15 + 1;
                goto LABEL_276;
              }
            }
LABEL_168:
            ++v15;
            v46 >>= 1;
            if ( v48 == v15 )
              goto LABEL_276;
          }
        }
      }
      if ( v18 )
        goto LABEL_5;
      v11 = *((_BYTE *)a2 + 24);
      if ( v11 == 2 )
      {
        v28 = a2[7];
        if ( *(_BYTE *)(v28 + 173) == 3 )
        {
          if ( (unsigned int)sub_8D2AC0(*(_QWORD *)(v28 + 128)) && (unsigned int)sub_8D2AC0(i) )
          {
            v104.m128i_i32[0] = 0;
            v105.m128i_i32[0] = 0;
            while ( *(_BYTE *)(i + 140) == 12 )
              i = *(_QWORD *)(i + 160);
            v29 = *(_BYTE *)(i + 160);
            sub_72A510(v28, v102);
            v30 = v102;
            *(_QWORD *)(v102 + 128) = i;
            v31 = v30 + 176;
            if ( (unsigned int)sub_70C5B0(v29, (unsigned int *)(v30 + 176)) )
            {
              v106[0].m128i_i64[0] = sub_724DC0(v29, v31, v32, v33, v34, v35);
              sub_72A510(v28, v106[0].m128i_i64[0]);
              sub_70BAF0(v29, (const __m128i *)(v106[0].m128i_i64[0] + 176), (_OWORD *)(v102 + 176), &v104, &v105);
              if ( v105.m128i_i32[0] )
                v104.m128i_i32[0] = 1;
              sub_724E30(v106);
            }
            if ( !v104.m128i_i32[0] )
              goto LABEL_63;
          }
          goto LABEL_5;
        }
      }
      goto LABEL_6;
    }
    if ( v17 <= 0x3D02u )
    {
      if ( v17 > 0x3CE8u )
      {
        switch ( v17 )
        {
          case 0x3CE9u:
          case 0x3CECu:
          case 0x3CEDu:
          case 0x3CEEu:
          case 0x3CF1u:
          case 0x3CF2u:
            goto LABEL_73;
          case 0x3CF4u:
          case 0x3CF8u:
          case 0x3D02u:
            goto LABEL_127;
          default:
            goto LABEL_4;
        }
      }
      v24 = v17 - 12474;
      if ( v24 > 0x13u || ((1LL << v24) & 0x80D03) == 0 )
        goto LABEL_5;
      v105.m128i_i32[0] = 0;
      if ( v18 || !sub_7175E0((__int64)a2, v106) || !(unsigned int)sub_8D2A90(i) )
      {
LABEL_4:
        if ( a2 )
          goto LABEL_5;
        goto LABEL_12;
      }
      v25 = *(_WORD *)(v9 + 176);
      sub_724C70(v102, 3);
      *(_QWORD *)(v102 + 128) = i;
      v26 = v106[0].m128i_i64[0];
      if ( *(_BYTE *)(v106[0].m128i_i64[0] + 173) == 6
        && (*(_BYTE *)(v106[0].m128i_i64[0] + 176) != 2
         || (v26 = *(_QWORD *)(v106[0].m128i_i64[0] + 184), *(_QWORD *)(v106[0].m128i_i64[0] + 192)))
        || *(_BYTE *)(v26 + 173) != 2
        || *(_QWORD *)(v26 + 176) != 1
        || (v27 = 0, **(_BYTE **)(v26 + 184)) )
      {
        v27 = sub_723880(*(char **)(v106[0].m128i_i64[0] + 184));
      }
      if ( v105.m128i_i32[0] )
        goto LABEL_5;
      v12 = sub_709F60(
              (__m128i *)(v102 + 176),
              *(_BYTE *)(i + 160),
              (v25 == 12493) | (unsigned __int8)((unsigned __int16)(v25 - 12484) <= 1u),
              v27);
LABEL_193:
      if ( v12 )
        goto LABEL_31;
      goto LABEL_4;
    }
    if ( v17 > 0x3D8Au )
    {
      if ( v17 != 16727 )
        goto LABEL_5;
      if ( dword_4F077C4 != 2
        && !v18
        && sub_7175E0((__int64)a2, v106)
        && (*(_BYTE *)(v106[0].m128i_i64[0] + 168) & 7) == 0
        && (unsigned int)sub_8D2780(i) )
      {
        v36 = *(_QWORD *)(v106[0].m128i_i64[0] + 176);
        if ( v36 )
        {
          v37 = 0;
          while ( *(_BYTE *)(*(_QWORD *)(v106[0].m128i_i64[0] + 184) + v37) )
          {
            if ( ++v37 == v36 )
              goto LABEL_5;
          }
          while ( *(_BYTE *)(i + 140) == 12 )
            i = *(_QWORD *)(i + 160);
          v12 = 1;
          sub_72BAF0(v102, v37, *(unsigned __int8 *)(i + 160));
          goto LABEL_31;
        }
        goto LABEL_5;
      }
      goto LABEL_4;
    }
    if ( v17 <= 0x3D87u )
      goto LABEL_5;
LABEL_117:
    v38 = v18 != 0;
LABEL_118:
    v93 = v38 || a2 == 0;
    if ( !v93 )
    {
      if ( (unsigned int)sub_8D2AC0(*a2) || (unsigned int)sub_8D3D40(*a2) )
      {
        v106[0].m128i_i32[0] = 0;
        v15 = *(_QWORD *)(v9 + 152);
        for ( jj = sub_73D790(v15); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
          ;
        v11 = *((_BYTE *)a2 + 24);
        if ( v11 == 2 )
        {
          v67 = (const __m128i *)a2[7];
          if ( v67[10].m128i_i8[13] == 3 )
          {
            for ( kk = v67[8].m128i_i64[0]; *(_BYTE *)(kk + 140) == 12; kk = *(_QWORD *)(kk + 160) )
              ;
            v69 = *(_WORD *)(v9 + 176);
            v70 = *(_BYTE *)(kk + 160);
            if ( v69 > 0x27F6u )
            {
              if ( (unsigned __int16)(v69 - 15752) <= 2u )
              {
                v86 = (unsigned int *)&v67[11];
                v98 = v70;
                if ( !(unsigned int)sub_709C40(v67 + 11, v70) )
                {
                  v82 = (int)sub_70C5B0(v98, v86);
LABEL_286:
                  if ( !v106[0].m128i_i32[0] )
                  {
                    v83 = *(unsigned __int8 *)(jj + 160);
                    v12 = 1;
                    sub_72BAF0(v102, v82, v83);
                    goto LABEL_31;
                  }
                }
                goto LABEL_5;
              }
            }
            else if ( v69 > 0x27E5u )
            {
              switch ( v69 )
              {
                case 0x27E6u:
                  v84 = v67 + 11;
                  v85 = v70;
                  if ( !sub_709CC0(v67 + 11, v70) )
                    v93 = (unsigned int)sub_709C40(v84, v85) == 0;
                  v82 = v93;
                  goto LABEL_286;
                case 0x27EBu:
                case 0x27EDu:
                case 0x27EEu:
                  v82 = sub_709CC0(v67 + 11, v70);
                  goto LABEL_286;
                case 0x27F3u:
                case 0x27F4u:
                case 0x27F5u:
                  v82 = (int)sub_709C40(v67 + 11, v70);
                  goto LABEL_286;
                case 0x27F6u:
                  v82 = (int)sub_709D20(v67 + 11, v70, v106);
                  goto LABEL_286;
                default:
                  break;
              }
            }
LABEL_175:
            sub_721090(v15);
          }
        }
      }
      else
      {
LABEL_121:
        *a5 = 1810;
        v11 = *((_BYTE *)a2 + 24);
      }
      goto LABEL_6;
    }
LABEL_53:
    *a5 = 1809;
    goto LABEL_4;
  }
  if ( v17 > 0x1264u )
  {
    if ( v17 <= 0x27F6u )
    {
      if ( v17 > 0x27C5u )
      {
        switch ( v17 )
        {
          case 0x27C6u:
          case 0x27C7u:
          case 0x27CEu:
            if ( !(unsigned int)sub_8D2A90(i) )
              goto LABEL_12;
            sub_724C70(v102, 3);
            v54 = (__m128i *)v102;
            *(_QWORD *)(v102 + 128) = i;
            v12 = sub_70A170(v54 + 11, *(_BYTE *)(i + 160));
            break;
          case 0x27E6u:
          case 0x27EBu:
          case 0x27EDu:
          case 0x27EEu:
          case 0x27F3u:
          case 0x27F4u:
          case 0x27F5u:
          case 0x27F6u:
            v38 = 0;
            goto LABEL_118;
          default:
            goto LABEL_12;
        }
      }
      else
      {
        if ( v17 <= 0x12C2u )
          goto LABEL_12;
        if ( v17 <= 0x12EEu )
        {
          if ( v17 <= 0x12ECu )
            goto LABEL_12;
        }
        else if ( v17 != 4853 )
        {
          goto LABEL_12;
        }
        if ( !(unsigned int)sub_8D2A90(i) )
          goto LABEL_12;
        sub_724C70(v102, 3);
        v21 = (__m128i *)v102;
        *(_QWORD *)(v102 + 128) = i;
        v12 = sub_70A1D0(v21 + 11, *(_BYTE *)(i + 160));
      }
      goto LABEL_193;
    }
    if ( v17 > 0x3D02u )
    {
      if ( v17 > 0x3D8Au || v17 <= 0x3D87u )
        goto LABEL_12;
      goto LABEL_53;
    }
    if ( v17 > 0x3CE8u )
    {
      switch ( v17 )
      {
        case 0x3CE9u:
        case 0x3CECu:
        case 0x3CEDu:
        case 0x3CEEu:
        case 0x3CF1u:
        case 0x3CF2u:
LABEL_154:
          v23 = 1;
          goto LABEL_155;
        case 0x3CF4u:
        case 0x3CF8u:
        case 0x3D02u:
          v18 = 0;
LABEL_127:
          if ( !dword_4F077C0 || qword_4F077A8 <= 0x76BFu )
            goto LABEL_4;
          if ( !a2 )
            goto LABEL_12;
          if ( !v18 )
            goto LABEL_4;
          v11 = *((_BYTE *)a2 + 24);
          if ( *(_QWORD *)(v18 + 16) || v11 != 2 || *(_BYTE *)(v18 + 24) != 2 )
            goto LABEL_6;
          v39 = *(const __m128i **)(v18 + 56);
          v100 = 0;
          v90 = v102;
          v92 = (const __m128i *)a2[7];
          for ( mm = v39[8].m128i_i64[0]; *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
            ;
          for ( nn = v92[8].m128i_i64[0]; *(_BYTE *)(nn + 140) == 12; nn = *(_QWORD *)(nn + 160) )
            ;
          v42 = v39 + 11;
          v88 = nn;
          v89 = mm;
          sub_70B720(*(_BYTE *)(mm + 160), v42, (__int64 *)&v103, &v100, &v101);
          if ( v100 )
            goto LABEL_5;
          if ( v103 >= 0x100 )
            goto LABEL_5;
          sub_70B680(*(_BYTE *)(v89 + 160), v103, v106, &v100);
          if ( (unsigned int)sub_70BE30(*(_BYTE *)(v89 + 160), v42, v106, &v100) )
            goto LABEL_5;
          if ( v100 )
            goto LABEL_5;
          sub_709EF0(v92 + 11, *(_BYTE *)(v88 + 160), &v104, 6u, &v100, &v101);
          if ( v100 )
            goto LABEL_5;
          sub_70B680(6u, 1, &v105, &v100);
          break;
        default:
          goto LABEL_12;
      }
      while ( 1 )
      {
        v43 = v103;
        if ( !v103 )
          break;
        if ( (v103 & 1) != 0 )
        {
          sub_70BBE0(6u, &v104, &v105, &v105, &v100, &v101);
          if ( v100 )
            goto LABEL_5;
          v43 = v103;
        }
        v103 = v43 / 2;
        if ( !(v43 / 2) )
          break;
        sub_70BBE0(6u, &v104, &v104, &v104, &v100, &v101);
        if ( v100 )
          goto LABEL_5;
      }
      sub_724C70(v90, 3);
      *(_QWORD *)(v90 + 128) = i;
      while ( *(_BYTE *)(i + 140) == 12 )
        i = *(_QWORD *)(i + 160);
      sub_709EF0(&v105, 6u, (_OWORD *)(v90 + 176), *(_BYTE *)(i + 160), &v100, &v101);
      if ( !v100 )
        goto LABEL_63;
      goto LABEL_5;
    }
  }
  else
  {
    if ( v17 > 0x1262u )
      goto LABEL_12;
    if ( v17 > 0x1177u )
    {
      if ( v17 > 0x11AFu || v17 <= 0x11ADu )
        goto LABEL_12;
LABEL_125:
      *a5 = 2539;
      goto LABEL_4;
    }
    if ( v17 > 0x112Au )
    {
      switch ( v17 )
      {
        case 0x112Bu:
        case 0x112Cu:
        case 0x1133u:
          v51 = 1;
LABEL_181:
          if ( !a2 )
            goto LABEL_12;
          if ( !v51 )
            goto LABEL_4;
          v11 = *((_BYTE *)a2 + 24);
          if ( v11 != 2 )
            goto LABEL_6;
          v52 = (__m128i *)a2[7];
          if ( v52[10].m128i_i8[13] != 3 )
            goto LABEL_6;
          if ( !(unsigned int)sub_8D2AC0(v52[8].m128i_i64[0]) || !(unsigned int)sub_8D2AC0(i) )
            goto LABEL_5;
          v106[0].m128i_i32[0] = 0;
          while ( *(_BYTE *)(i + 140) == 12 )
            i = *(_QWORD *)(i + 160);
          sub_72A510(v52, v102);
          v53 = (__m128i *)v102;
          *(_QWORD *)(v102 + 128) = i;
          sub_70C700(*(_BYTE *)(i + 160), v52 + 11, v53 + 11, v106);
          break;
        case 0x1150u:
          goto LABEL_175;
        case 0x1163u:
        case 0x1166u:
        case 0x1167u:
          goto LABEL_154;
        case 0x116Du:
LABEL_195:
          v12 = 1;
          sub_72BAF0(v102, *((_BYTE *)a2 + 24) == 2, *(unsigned __int8 *)(i + 160));
          goto LABEL_31;
        case 0x116Fu:
        case 0x1170u:
        case 0x1177u:
LABEL_179:
          *a5 = 3320;
          goto LABEL_4;
        default:
          goto LABEL_12;
      }
LABEL_62:
      if ( !v106[0].m128i_i32[0] )
      {
LABEL_63:
        v12 = 1;
LABEL_31:
        sub_72A510(v102, a4);
        goto LABEL_13;
      }
      goto LABEL_5;
    }
    if ( v17 == 4235 )
    {
      v18 = 0;
LABEL_26:
      v19 = v102;
      v11 = *((_BYTE *)a2 + 24);
      if ( v11 != 2 )
        goto LABEL_6;
      v20 = a2[7];
      if ( *(_BYTE *)(v20 + 173) != 1 )
      {
        v11 = 2;
        goto LABEL_6;
      }
      if ( *(_BYTE *)(v18 + 24) == 2 )
      {
        v80 = sub_620FD0(v20, v106);
        if ( ((v80 - 4) & 0xFFFFFFFFFFFFFFFBLL) == 0 || (unsigned __int64)(v80 - 1) <= 1 )
          goto LABEL_200;
      }
      else
      {
        v55 = sub_620FD0(v20, v106);
        if ( (unsigned __int64)(v55 - 1) <= 1 || v55 == 4 )
        {
LABEL_200:
          sub_72BBE0(v19, 1, *(unsigned __int8 *)(i + 160));
          goto LABEL_63;
        }
      }
      goto LABEL_5;
    }
    if ( v17 == 4174 )
    {
LABEL_38:
      *a5 = 2356;
      goto LABEL_4;
    }
  }
LABEL_12:
  v12 = 0;
LABEL_13:
  sub_724E30(&v102);
  return v12;
}
