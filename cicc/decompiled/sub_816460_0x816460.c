// Function: sub_816460
// Address: 0x816460
//
void __fastcall sub_816460(__int64 a1, unsigned int a2, unsigned int a3, __int64 *a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rax
  int v10; // eax
  const char *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  char *v14; // r15
  size_t v15; // rax
  char *v16; // rsi
  __int64 v17; // rdx
  char *v18; // rdx
  char v19; // al
  __int64 v20; // r15
  char *v21; // rsi
  _QWORD *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdi
  bool v30; // zf
  __int64 v31; // rax
  char v32; // bl
  unsigned __int64 i; // r14
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 j; // r12
  __int64 v38; // rdx
  char *v39; // rsi
  __int64 v40; // rbx
  __int64 v41; // rdi
  __int64 v42; // rdi
  unsigned int *v43; // rdi
  __int64 v44; // rax
  __int64 *v45; // rcx
  __int64 v46; // rdx
  _BOOL4 v47; // ecx
  const __m128i *v48; // rdi
  unsigned __int8 v49; // al
  __int64 v50; // rbx
  __int64 v51; // r8
  __int64 v52; // r15
  __int64 v53; // rax
  char v54; // dl
  char v55; // dl
  __int64 v56; // rax
  __int64 v57; // rsi
  unsigned int v58; // r12d
  char *v59; // rbx
  size_t v60; // rax
  __int64 v61; // r8
  _QWORD *v62; // rdi
  __int64 v63; // rax
  char v64; // al
  __int64 v65; // rdi
  int v66; // eax
  char *v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 *v71; // r8
  char v72; // al
  char *v73; // r15
  size_t v74; // rax
  char *v75; // r12
  size_t v76; // rax
  char v77; // al
  int v78; // ebx
  char *v79; // r15
  size_t v80; // rax
  char v81; // al
  __int64 v82; // rbx
  __int64 v83; // r15
  int v84; // esi
  char v85; // al
  __int64 v86; // rdi
  _QWORD *v87; // rdi
  __int64 v88; // rax
  _QWORD *v89; // rdi
  _QWORD *v90; // rdi
  _QWORD *v91; // r15
  __int64 v92; // [rsp+8h] [rbp-98h]
  __int64 v93; // [rsp+10h] [rbp-90h]
  __int64 v94; // [rsp+10h] [rbp-90h]
  unsigned int v95; // [rsp+1Ch] [rbp-84h] BYREF
  int v96; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 *v97; // [rsp+28h] [rbp-78h] BYREF
  __int64 v98[14]; // [rsp+30h] [rbp-70h] BYREF

  v95 = a3;
  v6 = (_QWORD *)sub_80AAC0(a1, &v95);
  if ( !v6 )
    return;
  v7 = v6;
  switch ( *((_BYTE *)v6 + 24) )
  {
    case 0:
      goto LABEL_4;
    case 1:
      v49 = *((_BYTE *)v6 + 56);
      if ( (unsigned __int8)(v49 - 105) > 4u )
      {
        if ( (unsigned __int8)(v49 - 22) > 1u )
        {
          if ( (unsigned __int8)(v49 - 100) > 1u && (unsigned __int8)(v49 - 94) > 1u )
          {
            switch ( v49 )
            {
              case 0u:
                v84 = 1;
                v85 = 11;
                goto LABEL_181;
              case 3u:
                v84 = 1;
                v85 = 7;
                goto LABEL_181;
              case 5u:
              case 6u:
              case 7u:
              case 0xAu:
              case 0xCu:
              case 0xEu:
              case 0xFu:
              case 0x10u:
              case 0x11u:
              case 0x14u:
                if ( dword_4D0425C )
                {
                  v78 = 1;
                  v79 = "cv";
                }
                else if ( (*((_BYTE *)v7 + 25) & 0x40) != 0 )
                {
                  v78 = 1;
                  v79 = "sc";
                }
                else
                {
                  v77 = *((_BYTE *)v7 + 58);
                  v78 = 1;
                  v79 = "cc";
                  if ( (v77 & 8) == 0 )
                  {
                    v79 = "cv";
                    if ( (v77 & 2) != 0 )
                      v79 = "rc";
                  }
                }
                goto LABEL_166;
              case 0x12u:
              case 0x13u:
                v79 = "dc";
                v78 = 1;
                if ( dword_4D0425C )
                  v79 = "cv";
                goto LABEL_166;
              case 0x18u:
                v78 = 0;
                v79 = "nx";
                goto LABEL_166;
              case 0x1Au:
                v84 = 1;
                v85 = 6;
                goto LABEL_181;
              case 0x1Bu:
                v84 = 1;
                v85 = 5;
                goto LABEL_181;
              case 0x1Cu:
              case 0x20u:
                v84 = 1;
                v85 = 13;
                goto LABEL_181;
              case 0x1Du:
              case 0x1Eu:
                v84 = 1;
                v85 = 14;
                goto LABEL_181;
              case 0x21u:
                v78 = 0;
                v79 = "v18__real__";
                goto LABEL_166;
              case 0x22u:
                v78 = 0;
                v79 = "v18__imag__";
                goto LABEL_166;
              case 0x23u:
              case 0x25u:
                v84 = 2;
                v85 = 37;
                goto LABEL_181;
              case 0x24u:
              case 0x26u:
                v84 = 2;
                v85 = 38;
                goto LABEL_181;
              case 0x27u:
              case 0x32u:
                v84 = 2;
                v85 = 5;
                goto LABEL_181;
              case 0x28u:
              case 0x33u:
              case 0x34u:
                v84 = 2;
                v85 = 6;
                goto LABEL_181;
              case 0x29u:
                v84 = 2;
                v85 = 7;
                goto LABEL_181;
              case 0x2Au:
                v84 = 2;
                v85 = 8;
                goto LABEL_181;
              case 0x2Bu:
                v84 = 2;
                v85 = 9;
                goto LABEL_181;
              case 0x35u:
                v84 = 2;
                v85 = 26;
                goto LABEL_181;
              case 0x36u:
                v84 = 2;
                v85 = 27;
                goto LABEL_181;
              case 0x37u:
                v84 = 2;
                v85 = 11;
                goto LABEL_181;
              case 0x38u:
                v84 = 2;
                v85 = 12;
                goto LABEL_181;
              case 0x39u:
                v84 = 2;
                v85 = 10;
                goto LABEL_181;
              case 0x3Au:
              case 0x41u:
                v84 = 2;
                v85 = 30;
                goto LABEL_181;
              case 0x3Bu:
              case 0x42u:
                v84 = 2;
                v85 = 31;
                goto LABEL_181;
              case 0x3Cu:
              case 0x43u:
                v84 = 2;
                v85 = 17;
                goto LABEL_181;
              case 0x3Du:
              case 0x44u:
                v84 = 2;
                v85 = 16;
                goto LABEL_181;
              case 0x3Eu:
              case 0x45u:
                v84 = 2;
                v85 = 33;
                goto LABEL_181;
              case 0x3Fu:
              case 0x46u:
                v84 = 2;
                v85 = 32;
                goto LABEL_181;
              case 0x40u:
                v84 = 2;
                v85 = 34;
                goto LABEL_181;
              case 0x47u:
                v84 = 2;
                v85 = 45;
                goto LABEL_181;
              case 0x48u:
                v84 = 2;
                v85 = 46;
                goto LABEL_181;
              case 0x49u:
                v84 = 2;
                v85 = 15;
                goto LABEL_181;
              case 0x4Au:
                v84 = 2;
                v85 = 18;
                goto LABEL_181;
              case 0x4Bu:
                v84 = 2;
                v85 = 19;
                goto LABEL_181;
              case 0x4Cu:
                v84 = 2;
                v85 = 20;
                goto LABEL_181;
              case 0x4Du:
                v84 = 2;
                v85 = 21;
                goto LABEL_181;
              case 0x4Eu:
                v84 = 2;
                v85 = 22;
                goto LABEL_181;
              case 0x4Fu:
                v84 = 2;
                v85 = 29;
                goto LABEL_181;
              case 0x50u:
                v84 = 2;
                v85 = 28;
                goto LABEL_181;
              case 0x51u:
                v84 = 2;
                v85 = 24;
                goto LABEL_181;
              case 0x52u:
                v84 = 2;
                v85 = 25;
                goto LABEL_181;
              case 0x53u:
                v84 = 2;
                v85 = 23;
                goto LABEL_181;
              case 0x57u:
              case 0x59u:
                v84 = 2;
                v85 = 35;
                goto LABEL_181;
              case 0x58u:
              case 0x5Au:
                v84 = 2;
                v85 = 36;
                goto LABEL_181;
              case 0x5Bu:
                v84 = 2;
                v85 = 39;
                goto LABEL_181;
              case 0x5Cu:
                v84 = 2;
                v85 = 43;
                goto LABEL_181;
              case 0x60u:
              case 0x62u:
                v78 = 0;
                v79 = "ds";
                goto LABEL_166;
              case 0x61u:
              case 0x63u:
                v84 = 2;
                v85 = 40;
                goto LABEL_181;
              case 0x67u:
              case 0x68u:
                v84 = 3;
                v85 = 44;
LABEL_181:
                v78 = 0;
                v79 = sub_8094C0(v85, v84);
LABEL_166:
                v80 = strlen(v79);
                *a4 += v80;
                sub_8238B0(qword_4F18BE0, v79, v80);
                if ( (unsigned __int8)(*((_BYTE *)v7 + 56) - 37) <= 1u )
                {
                  v90 = (_QWORD *)qword_4F18BE0;
                  ++*a4;
                  if ( (unsigned __int64)(v90[2] + 1LL) > v90[1] )
                  {
                    sub_823810(v90);
                    v90 = (_QWORD *)qword_4F18BE0;
                  }
                  *(_BYTE *)(v90[4] + v90[2]++) = 95;
                }
                if ( v78 )
                {
                  v81 = *((_BYTE *)v7 + 56);
                  if ( v81 == 7 || v81 == 19 )
                  {
                    v91 = sub_7259C0(6);
                    sub_737520(v7, (__int64)v91);
                    sub_8163E0((__int64)v91, a4);
                  }
                  else
                  {
                    sub_8163E0(*v7, a4);
                  }
                }
                v82 = v7[9];
                if ( v82 )
                {
                  v83 = 0;
                  do
                  {
                    if ( (*((_BYTE *)v7 + 59) & 0x10) != 0 && *(_QWORD *)(v7[9] + 16LL) == v82 )
                      sub_80E120(a4);
                    else
                      sub_816460(v82, a2, 0, a4);
                    if ( *((_BYTE *)v7 + 56) == 110 && (unsigned __int64)++v83 > 8 )
                      break;
                    v82 = *(_QWORD *)(v82 + 16);
                  }
                  while ( v82 );
                }
                return;
              default:
                goto LABEL_8;
            }
          }
          sub_8187D0(v7, 0, a2, a4);
          return;
        }
        v50 = v7[9];
LABEL_102:
        if ( v49 > 0x69u )
        {
          v51 = *(_QWORD *)(v50 + 16);
          v52 = *(_QWORD *)(v51 + 16);
          goto LABEL_80;
        }
        if ( v49 > 0x17u )
        {
          if ( v49 == 105 )
          {
            v52 = *(_QWORD *)(v50 + 16);
            v51 = 0;
            goto LABEL_80;
          }
        }
        else if ( v49 > 0x15u )
        {
          v51 = 0;
          v52 = 0;
          goto LABEL_80;
        }
LABEL_8:
        sub_721090();
      }
      v50 = v7[9];
      if ( v49 <= 0x6Bu )
        goto LABEL_102;
      v51 = 0;
      v52 = *(_QWORD *)(*(_QWORD *)(v50 + 16) + 16LL);
LABEL_80:
      v92 = v51;
      v53 = sub_72B0F0(v50, 0);
      v54 = *((_BYTE *)v7 + 59);
      v93 = v53;
      if ( (v54 & 8) == 0 || *(_BYTE *)(v53 + 174) == 4 )
      {
        v65 = qword_4F18BE0;
        *a4 += 2;
        if ( (v54 & 2) != 0 )
          sub_8238B0(v65, "cp", 2);
        else
          sub_8238B0(v65, "cl", 2);
        if ( *((_BYTE *)v7 + 56) == 105 )
        {
          if ( (*((_BYTE *)v7 + 59) & 8) != 0 && *(_BYTE *)(v93 + 174) == 4 )
          {
            *a4 += 3;
            sub_8238B0(qword_4F18BE0, &unk_3C1BC3F, 3);
            sub_8111C0(v93, 0, 1, 1, 0, 0, (__int64)a4);
            v89 = (_QWORD *)qword_4F18BE0;
            ++*a4;
            if ( (unsigned __int64)(v89[2] + 1LL) > v89[1] )
            {
              sub_823810(v89);
              v89 = (_QWORD *)qword_4F18BE0;
            }
            *(_BYTE *)(v89[4] + v89[2]++) = 69;
          }
          else
          {
            sub_817D30(v50, v52, 0, a2, a4);
          }
        }
        else
        {
          sub_8187D0(v7, v52, a2, a4);
        }
        sub_817850(v52, a2, a4);
        goto LABEL_97;
      }
      v55 = *(_BYTE *)(v53 + 176);
      if ( !v52 )
      {
        v57 = 0;
        if ( !v92 )
        {
          v58 = 0;
          if ( (unsigned __int8)(v55 - 37) > 1u )
            goto LABEL_89;
          goto LABEL_157;
        }
        goto LABEL_87;
      }
      v56 = v52;
      v57 = 0;
      do
      {
        if ( (*(_BYTE *)(v56 + 25) & 0x10) != 0 )
          break;
        v56 = *(_QWORD *)(v56 + 16);
        ++v57;
      }
      while ( v56 );
      if ( v92 )
LABEL_87:
        ++v57;
      v58 = 0;
      if ( (unsigned __int8)(v55 - 37) > 1u )
      {
LABEL_89:
        v59 = sub_8094C0(v55, v57);
        v60 = strlen(v59);
        *a4 += v60;
        sub_8238B0(qword_4F18BE0, v59, v60);
        v61 = v92;
        goto LABEL_90;
      }
      if ( v57 != 1 )
      {
LABEL_157:
        LODWORD(v57) = v57 - 1;
        v58 = 1;
        goto LABEL_89;
      }
      v75 = sub_8094C0(v55, 1);
      v76 = strlen(v75);
      *a4 += v76;
      sub_8238B0(qword_4F18BE0, v75, v76);
      v61 = v92;
      v58 = dword_4D0425C;
      if ( dword_4D0425C )
      {
        v58 = 0;
      }
      else
      {
        v87 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        v88 = v87[2];
        if ( (unsigned __int64)(v88 + 1) > v87[1] )
        {
          sub_823810(v87);
          v87 = (_QWORD *)qword_4F18BE0;
          v61 = v92;
          v88 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v87[4] + v88) = 95;
        ++v87[2];
      }
LABEL_90:
      if ( v61 )
        sub_816460(v61, a2, 0, a4);
      for ( ; v52; v52 = *(_QWORD *)(v52 + 16) )
      {
        if ( v58 && !*(_QWORD *)(v52 + 16) )
          break;
        sub_816460(v52, a2, 0, a4);
      }
      if ( *(_BYTE *)(v93 + 176) == 42 )
      {
LABEL_97:
        v62 = (_QWORD *)qword_4F18BE0;
        ++*a4;
        v63 = v62[2];
        if ( (unsigned __int64)(v63 + 1) > v62[1] )
        {
          sub_823810(v62);
          v62 = (_QWORD *)qword_4F18BE0;
          v63 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v62[4] + v63) = 69;
        ++v62[2];
      }
      return;
    case 2:
      v48 = (const __m128i *)v6[7];
      if ( v48[10].m128i_i8[13] == 12 && v48[11].m128i_i8[0] == 9 && !v95 )
        sub_8156E0((__int64)v48, 2u, 0, 1, a4);
      else
        sub_80D8A0(v48, a2, v95, a4);
      return;
    case 3:
      v41 = v6[7];
      if ( (*(_BYTE *)(v41 + 172) & 1) != 0 )
      {
        *a4 += 3;
        sub_8238B0(qword_4F18BE0, "fpT", 3);
      }
      else
      {
        sub_8156E0(v41, 7u, 0, 0, a4);
      }
      return;
    case 4:
      sub_8156E0(v6[7], 8u, 0, 0, a4);
      return;
    case 5:
      sub_817A40(v6[7], *v6, (*((_BYTE *)v6 + 25) & 0x40) != 0, a4);
      return;
    case 6:
      sub_817960(0, 0, *v6, a4);
      return;
    case 7:
      v18 = (char *)v6[7];
      v19 = *v18;
      v20 = *((_QWORD *)v18 + 3);
      if ( (*v18 & 0x10) != 0 )
      {
        *a4 += 2;
        sub_8238B0(qword_4F18BE0, "gs", 2);
        v18 = (char *)v7[7];
        v19 = *v18;
      }
      if ( (v19 & 1) != 0 )
      {
        v21 = "na";
        if ( !(unsigned int)sub_8D3410(*((_QWORD *)v18 + 1)) )
          v21 = "nw";
      }
      else
      {
        v21 = "da";
        if ( (v19 & 8) == 0 )
          v21 = "dl";
      }
      *a4 += 2;
      sub_8238B0(qword_4F18BE0, v21, 2);
      if ( v20 )
        sub_817850(v20, a2, a4);
      if ( (*(_BYTE *)v7[7] & 1) == 0 )
        return;
      v22 = (_QWORD *)qword_4F18BE0;
      ++*a4;
      v23 = v22[2];
      if ( (unsigned __int64)(v23 + 1) > v22[1] )
      {
        sub_823810(v22);
        v22 = (_QWORD *)qword_4F18BE0;
        v23 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v22[4] + v23) = 95;
      ++v22[2];
      sub_80F5E0(*(_QWORD *)(v7[7] + 8LL), 0, a4);
      v24 = *(_QWORD *)(v7[7] + 32LL);
      if ( !v24 )
        goto LABEL_97;
      while ( 1 )
      {
        v25 = sub_730290(v24);
        if ( (*(_BYTE *)(v25 + 51) & 0x40) != 0 )
          v25 = sub_730770(v25, 0);
        if ( v25 == v24 )
          break;
        v24 = v25;
      }
      if ( (*(_BYTE *)(v25 + 50) & 0x40) != 0 )
      {
        sub_809D10(v25, (__int64 *)&v97, v98);
        sub_817960(v97, v98[0], 0, a4);
        return;
      }
      *a4 += 2;
      v94 = v25;
      sub_8238B0(qword_4F18BE0, "pi", 2);
      sub_809D10(v94, (__int64 *)&v97, v98);
      sub_8178D0(v97, v98[0], a4);
      goto LABEL_97;
    case 8:
      v29 = qword_4F18BE0;
      v30 = v6[7] == 0;
      *a4 += 2;
      if ( v30 )
      {
        sub_8238B0(v29, "tr", 2);
      }
      else
      {
        sub_8238B0(v29, "tw", 2);
        sub_817A40(*(_QWORD *)(v7[7] + 8LL), *(_QWORD *)v7[7], 0, a4);
      }
      return;
    case 0xB:
      v26 = v6[7];
      v27 = 0;
      v28 = *(_QWORD *)(v26 + 16);
      if ( !v28 )
        v27 = *(_QWORD *)(v26 + 56);
      sub_818B40(v27, v28, 9, a4);
      return;
    case 0xC:
      v44 = v6[8];
      v45 = a4;
      v46 = 5;
      if ( *((_BYTE *)v7 + 56) )
        goto LABEL_68;
      goto LABEL_74;
    case 0xD:
      *a4 += 2;
      LODWORD(v98[0]) = 0;
      sub_8238B0(qword_4F18BE0, "sZ", 2);
      v42 = v7[8];
      if ( *((_BYTE *)v7 + 57) )
      {
        v43 = (unsigned int *)(v42 + 128);
      }
      else
      {
        if ( !*((_BYTE *)v7 + 56) )
        {
          v70 = sub_80AAC0(v42, v98);
          v71 = (__int64 *)v70;
          v72 = *(_BYTE *)(v70 + 24);
          if ( v72 == 2 )
          {
            v86 = v71[7];
            if ( *(_BYTE *)(v86 + 173) == 12 && !*(_BYTE *)(v86 + 176) )
            {
              v43 = (unsigned int *)(v86 + 184);
              goto LABEL_66;
            }
          }
          else if ( v72 == 24 && *((_DWORD *)v71 + 14) )
          {
            sub_80C310(v71, a4);
            return;
          }
          sub_816460(v71, 0, 0, a4);
          return;
        }
        while ( 1 )
        {
          v64 = *(_BYTE *)(v42 + 140);
          if ( v64 != 12 )
            break;
          v42 = *(_QWORD *)(v42 + 160);
        }
        if ( v64 != 14 || *(_BYTE *)(v42 + 160) )
        {
          sub_80BC40("?", a4);
          return;
        }
        v43 = (unsigned int *)(*(_QWORD *)(v42 + 168) + 24LL);
      }
LABEL_66:
      sub_812B60(v43, 0, a4);
      return;
    case 0xE:
      v44 = v6[8];
      v45 = a4;
      v46 = 7;
      if ( *((_BYTE *)v7 + 56) )
LABEL_68:
        sub_818B40(v44, 0, v46, v45);
      else
LABEL_74:
        sub_818B40(0, v44, v46, v45);
      return;
    case 0x12:
      sub_80E120(a4);
      return;
    case 0x14:
      v47 = 0;
      if ( !v95 )
        v47 = (*((_BYTE *)v6 + 25) & 1) == 0;
      sub_8156E0(v6[7], 0xBu, 0, v47, a4);
      return;
    case 0x17:
      v31 = v6[8];
      v32 = *((_BYTE *)v7 + 56);
      for ( i = 0; v31; ++i )
      {
        if ( (*(_BYTE *)(v31 + 25) & 0x10) != 0 )
          break;
        v31 = *(_QWORD *)(v31 + 16);
      }
      v34 = (_QWORD *)qword_4F18BE0;
      ++*a4;
      v35 = v34[2];
      if ( (unsigned __int64)(v35 + 1) > v34[1] )
      {
        sub_823810(v34);
        v34 = (_QWORD *)qword_4F18BE0;
        v35 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v34[4] + v35) = 118;
      ++v34[2];
      if ( i > 9 )
      {
        v66 = sub_622470(i, v98);
        v34 = (_QWORD *)qword_4F18BE0;
        v36 = v66;
      }
      else
      {
        v36 = 1;
        LOWORD(v98[0]) = (unsigned __int8)(i + 48);
      }
      *a4 += v36;
      sub_8238B0(v34, v98, v36);
      if ( (unsigned __int8)v32 > 0x63u )
      {
        *a4 += 9;
        sub_8238B0(qword_4F18BE0, "10builtin", 9);
        ++*a4;
        LOWORD(v98[0]) = (unsigned __int8)((unsigned __int8)v32 / 0x64u + 48);
        sub_8238B0(qword_4F18BE0, v98, 1);
        v32 = (unsigned __int8)v32 % 0x64u;
      }
      else
      {
        *a4 += 8;
        sub_8238B0(qword_4F18BE0, "9builtin", 8);
      }
      ++*a4;
      LOWORD(v98[0]) = (unsigned __int8)(v32 / 10 + 48);
      sub_8238B0(qword_4F18BE0, v98, 1);
      ++*a4;
      LOWORD(v98[0]) = (unsigned __int8)(v32 % 10 + 48);
      sub_8238B0(qword_4F18BE0, v98, 1);
      for ( j = v7[8]; j; j = *(_QWORD *)(j + 16) )
      {
        if ( *(_BYTE *)(j + 24) == 22 )
        {
          *a4 += 2;
          sub_8238B0(qword_4F18BE0, "TO", 2);
          sub_80F5E0(*(_QWORD *)(j + 56), 0, a4);
        }
        else
        {
          sub_816460(j, 1, 0, a4);
        }
      }
      return;
    case 0x18:
      sub_80C310(v6, a4);
      return;
    case 0x19:
      sub_817960(v6[7], 0, 0, a4);
      return;
    case 0x1D:
      *a4 += 2;
      v38 = 2;
      v39 = "aw";
      goto LABEL_56;
    case 0x1E:
      v10 = *((_BYTE *)v6 + 66) & 1;
      if ( *(_QWORD *)(v7[7] + 16LL) )
      {
        v11 = "fL";
        v12 = qword_4F18BE0;
        if ( !v10 )
          v11 = "fR";
        *a4 += 2;
        sub_8238B0(v12, v11, 2);
        v13 = *((unsigned __int16 *)v7 + 32);
        if ( (_WORD)v13 == 147 )
        {
          *a4 += 2;
          v17 = 2;
          v16 = "ds";
        }
        else
        {
          v14 = sub_8094C0(byte_4B6D300[v13], 2);
          v15 = strlen(v14);
          *a4 += v15;
          v16 = v14;
          v17 = v15;
        }
        sub_8238B0(qword_4F18BE0, v16, v17);
        sub_816460(v7[7], a2, 0, a4);
        sub_816460(*(_QWORD *)(v7[7] + 16LL), a2, 0, a4);
      }
      else
      {
        v67 = "fl";
        v68 = qword_4F18BE0;
        if ( !v10 )
          v67 = "fr";
        *a4 += 2;
        sub_8238B0(v68, v67, 2);
        v69 = *((unsigned __int16 *)v7 + 32);
        if ( (_WORD)v69 == 147 )
        {
          *a4 += 2;
          v38 = 2;
          v39 = "ds";
        }
        else
        {
          v73 = sub_8094C0(byte_4B6D300[v69], 2);
          v74 = strlen(v73);
          v39 = v73;
          *a4 += v74;
          v38 = v74;
        }
LABEL_56:
        sub_8238B0(qword_4F18BE0, v39, v38);
        sub_816460(v7[7], a2, 0, a4);
      }
      return;
    case 0x20:
      v96 = 0;
      v40 = v6[7];
      if ( HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 )
      {
        v97 = 0;
        sub_80BC40(*(char **)(v40 + 8), a4);
        v98[0] = v7[8];
        sub_811CB0(v98, 0, 0, a4);
        sub_80C110(v96, v97, a4);
        return;
      }
      *a4 += 3;
      v97 = 0;
      sub_8238B0(qword_4F18BE0, &unk_3C1BC3F, 3);
      if ( !(unsigned int)sub_80C5A0(v7[7], 59, 0, 0, v98, a4) )
      {
        sub_811730(v40, 0x3Bu, &v96, (__int64 *)&v97, 0, (__int64)a4);
        sub_80BC40(*(char **)(v40 + 8), a4);
        if ( !a4[5] )
          sub_80A250(v7[7], 59, 0, (__int64)a4);
      }
      v98[0] = v7[8];
      sub_811CB0(v98, 0, 0, a4);
      sub_80C110(v96, v97, a4);
      goto LABEL_97;
    case 0x21:
      sub_684AA0(7u, 0xCBEu, (_DWORD *)v6 + 7);
      sub_67D850(3262, 1, 0);
LABEL_4:
      v8 = (_QWORD *)qword_4F18BE0;
      ++*a4;
      v9 = v8[2];
      if ( (unsigned __int64)(v9 + 1) > v8[1] )
      {
        sub_823810(v8);
        v8 = (_QWORD *)qword_4F18BE0;
        v9 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v8[4] + v9) = 63;
      ++v8[2];
      return;
    default:
      goto LABEL_8;
  }
}
