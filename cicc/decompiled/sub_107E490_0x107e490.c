// Function: sub_107E490
// Address: 0x107e490
//
__int64 *__fastcall sub_107E490(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 *result; // rax
  __int64 *i; // rbx
  int v10; // eax
  __int64 v11; // r12
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // r8
  _QWORD *v15; // r9
  int v16; // r11d
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rdi
  int v24; // eax
  unsigned int v25; // esi
  __int64 v26; // r11
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // r9
  int v30; // r8d
  char v31; // al
  __int64 *v32; // rdi
  char v33; // cl
  __int64 v34; // rdx
  int v35; // eax
  int v36; // r9d
  __int64 *v37; // rdx
  int v38; // r8d
  char v39; // al
  __int64 v40; // rdx
  int v41; // eax
  int v42; // r9d
  _BYTE *v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // rdi
  unsigned __int64 v46; // r10
  unsigned int v47; // r8d
  __int64 v48; // r11
  int v49; // r8d
  char v50; // al
  __int64 v51; // rdx
  int v52; // r9d
  unsigned __int64 v53; // r10
  unsigned int v54; // r8d
  __int64 v55; // r11
  __int64 v56; // rdx
  unsigned int v57; // r8d
  __int64 v58; // r10
  unsigned int v59; // esi
  __int64 v60; // rcx
  __int64 v61; // r9
  _BYTE *v62; // rdi
  __int64 v63; // rdi
  unsigned int v64; // ecx
  __int64 v65; // r9
  _QWORD *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdi
  int v69; // eax
  int v70; // edx
  __int64 v71; // rax
  int v72; // esi
  __int64 v73; // r8
  int v74; // eax
  int v75; // edx
  __int64 v76; // rax
  int v77; // [rsp+4h] [rbp-6Ch]
  int v78; // [rsp+4h] [rbp-6Ch]
  __int64 v79; // [rsp+8h] [rbp-68h]
  __int64 v81; // [rsp+18h] [rbp-58h]
  __int64 v82; // [rsp+18h] [rbp-58h]
  __int64 *v84; // [rsp+28h] [rbp-48h]
  __int64 v85[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = a1 + 296;
  v7 = **(_QWORD **)(v6 - 192);
  v79 = v6;
  result = &a2[5 * a3];
  v84 = result;
  if ( a2 != result )
  {
    for ( i = a2 + 1; ; i = result )
    {
      v10 = *((_DWORD *)i + 4);
      v11 = *(_QWORD *)(i[3] + 160) + *(i - 1) + a4;
      if ( v10 == 7 || v10 == 13 )
      {
        v56 = *i;
        if ( !*(_BYTE *)(*i + 36) || *(_DWORD *)(v56 + 32) != 2 )
          break;
      }
      switch ( v10 )
      {
        case 0:
        case 7:
        case 10:
        case 13:
        case 20:
        case 26:
          v25 = *(_DWORD *)(a1 + 256);
          v81 = a1 + 232;
          if ( !v25 )
          {
            ++*(_QWORD *)(a1 + 232);
            v85[0] = 0;
LABEL_113:
            v25 *= 2;
            goto LABEL_114;
          }
          v26 = *(_QWORD *)(a1 + 240);
          v27 = (v25 - 1) & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
          v28 = v26 + 16LL * v27;
          v29 = *(_QWORD *)v28;
          if ( *i == *(_QWORD *)v28 )
          {
LABEL_16:
            v22 = *(unsigned int *)(v28 + 8);
            break;
          }
          v78 = 1;
          v73 = 0;
          while ( v29 != -4096 )
          {
            if ( !v73 && v29 == -8192 )
              v73 = v28;
            v27 = (v25 - 1) & (v78 + v27);
            v28 = v26 + 16LL * v27;
            v29 = *(_QWORD *)v28;
            if ( *i == *(_QWORD *)v28 )
              goto LABEL_16;
            ++v78;
          }
          v74 = *(_DWORD *)(a1 + 248);
          if ( !v73 )
            v73 = v28;
          ++*(_QWORD *)(a1 + 232);
          v75 = v74 + 1;
          v85[0] = v73;
          if ( 4 * (v74 + 1) >= 3 * v25 )
            goto LABEL_113;
          if ( v25 - *(_DWORD *)(a1 + 252) - v75 > v25 >> 3 )
            goto LABEL_109;
LABEL_114:
          sub_107DA80(v81, v25);
          sub_107C950(v81, i, v85);
          v73 = v85[0];
          v75 = *(_DWORD *)(a1 + 248) + 1;
LABEL_109:
          *(_DWORD *)(a1 + 248) = v75;
          if ( *(_QWORD *)v73 != -4096 )
            --*(_DWORD *)(a1 + 252);
          v76 = *i;
          v22 = 0;
          *(_DWORD *)(v73 + 8) = 0;
          *(_QWORD *)v73 = v76;
          v10 = *((_DWORD *)i + 4);
          break;
        case 1:
        case 2:
        case 12:
        case 18:
        case 19:
        case 24:
          v23 = a1 + 200;
          v85[0] = sub_E5C930(a5, *i);
          v24 = *((_DWORD *)i + 4);
          if ( v24 == 12 || v24 == 24 )
            v22 = (unsigned int)(*(_DWORD *)sub_107DEB0(v23, v85) - 1);
          else
            v22 = *(unsigned int *)sub_107DEB0(v23, v85);
          goto LABEL_50;
        case 3:
        case 4:
        case 5:
        case 11:
        case 14:
        case 15:
        case 16:
        case 17:
        case 21:
        case 23:
        case 25:
          if ( !sub_1079CD0(*i, 1) )
            goto LABEL_86;
          v12 = *(_DWORD *)(a1 + 320);
          if ( v12 )
          {
            v13 = v12 - 1;
            v14 = *(_QWORD *)(a1 + 304);
            v15 = 0;
            v16 = 1;
            v17 = v13 & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
            v18 = (__int64 *)(v14 + 32LL * v17);
            v19 = *v18;
            if ( *v18 == *i )
            {
LABEL_9:
              v20 = v18[2];
              v21 = 80LL * *((unsigned int *)v18 + 2);
              goto LABEL_10;
            }
            while ( v19 != -4096 )
            {
              if ( v19 == -8192 && !v15 )
                v15 = v18;
              v17 = v13 & (v16 + v17);
              v18 = (__int64 *)(v14 + 32LL * v17);
              v19 = *v18;
              if ( *i == *v18 )
                goto LABEL_9;
              ++v16;
            }
            if ( !v15 )
              v15 = v18;
          }
          else
          {
            v15 = 0;
          }
          v66 = sub_107E2F0(v79, i, v15);
          v67 = *i;
          v21 = 0;
          *((_DWORD *)v66 + 2) = 0;
          *v66 = v67;
          v20 = 0;
          v66[2] = 0;
          v66[3] = 0;
LABEL_10:
          v10 = *((_DWORD *)i + 4);
          v22 = i[1] + v20 + *(_QWORD *)(*(_QWORD *)(a1 + 736) + v21 + 32);
          break;
        case 6:
          v22 = (unsigned int)sub_1078680(a1, *i, 6);
          v10 = *((_DWORD *)i + 4);
          break;
        case 8:
        case 9:
        case 22:
          if ( sub_1079CD0(*i, 1) )
          {
            v22 = i[1] + *(_QWORD *)(*((_QWORD *)sub_1079CD0(*i, 1) + 1) + 160LL);
LABEL_50:
            v10 = *((_DWORD *)i + 4);
          }
          else
          {
LABEL_86:
            v10 = *((_DWORD *)i + 4);
            v22 = 0;
          }
          break;
        default:
LABEL_142:
          BUG();
      }
LABEL_17:
      switch ( v10 )
      {
        case 0:
        case 3:
        case 6:
        case 7:
        case 10:
        case 20:
          sub_1076D00(v7, v22, v11);
          result = i + 5;
          if ( v84 == i + 4 )
            return result;
          continue;
        case 1:
        case 4:
        case 11:
        case 12:
        case 21:
          v30 = 0;
          v31 = v22;
          v32 = v85;
          v33 = v22 & 0x7F;
          v34 = (__int64)(int)v22 >> 7;
          if ( !v34 )
            goto LABEL_24;
          while ( v34 != -1 || (v31 & 0x40) == 0 )
          {
            while ( 1 )
            {
              v32 = (__int64 *)((char *)v32 + 1);
              ++v30;
              v31 = v34;
              *((_BYTE *)v32 - 1) = v33 | 0x80;
              v33 = v34 & 0x7F;
              v34 >>= 7;
              if ( v34 )
                break;
LABEL_24:
              v35 = v31 & 0x40;
              if ( !v35 )
              {
                v36 = v30 + 1;
                v37 = (__int64 *)((char *)v32 + 1);
                if ( (unsigned int)(v30 + 1) > 4 )
                  goto LABEL_29;
                *(_BYTE *)v32 = v33 | 0x80;
                v62 = (char *)v32 + 1;
                goto LABEL_74;
              }
            }
          }
          v36 = v30 + 1;
          v37 = (__int64 *)((char *)v32 + 1);
          if ( (unsigned int)(v30 + 1) > 4 )
            goto LABEL_29;
          LOBYTE(v35) = 127;
          *(_BYTE *)v32 = v33 | 0x80;
          v62 = (char *)v32 + 1;
LABEL_74:
          if ( v36 != 4 )
          {
            v63 = (unsigned int)(3 - v30);
            if ( (unsigned int)(v30 + 2) > 4 )
              v63 = 1;
            if ( (_DWORD)v63 )
            {
              v64 = 0;
              do
              {
                v65 = v64++;
                *((_BYTE *)v37 + v65) = v35 | 0x80;
              }
              while ( v64 < (unsigned int)v63 );
            }
            v62 = (char *)v37 + v63;
          }
          *v62 = v35;
          LODWORD(v37) = (_DWORD)v62 + 1;
          goto LABEL_30;
        case 2:
        case 5:
        case 8:
        case 9:
        case 13:
        case 23:
        case 26:
          LODWORD(v85[0]) = v22;
          (*(void (__fastcall **)(__int64, __int64 *, __int64, __int64))(*(_QWORD *)v7 + 104LL))(v7, v85, 4, v11);
          goto LABEL_19;
        case 14:
          v49 = 0;
          v50 = v22;
          v32 = v85;
          v33 = v22 & 0x7F;
          v51 = v22 >> 7;
          if ( !(v22 >> 7) )
            goto LABEL_52;
          while ( v51 != -1 || (v50 & 0x40) == 0 )
          {
            while ( 1 )
            {
              v32 = (__int64 *)((char *)v32 + 1);
              ++v49;
              v50 = v51;
              *((_BYTE *)v32 - 1) = v33 | 0x80;
              v33 = v51 & 0x7F;
              v51 >>= 7;
              if ( v51 )
                break;
LABEL_52:
              v41 = v50 & 0x40;
              if ( !v41 )
              {
                v52 = v49 + 1;
                v37 = (__int64 *)((char *)v32 + 1);
                if ( (unsigned int)(v49 + 1) > 9 )
                  goto LABEL_29;
                *(_BYTE *)v32 = v33 | 0x80;
                v43 = (char *)v32 + 1;
                goto LABEL_58;
              }
            }
          }
          v52 = v49 + 1;
          v37 = (__int64 *)((char *)v32 + 1);
          if ( (unsigned int)(v49 + 1) > 9 )
            goto LABEL_29;
          LOBYTE(v41) = 127;
          *(_BYTE *)v32 = v33 | 0x80;
          v43 = (char *)v32 + 1;
LABEL_58:
          if ( v52 == 9 )
            goto LABEL_46;
          v44 = (unsigned int)(8 - v49);
          if ( (unsigned int)(v49 + 2) > 9 )
            v44 = 1;
          v45 = 0x101010101010101LL * ((unsigned __int8)v41 | 0x80u);
          if ( (unsigned int)v44 < 8 )
            goto LABEL_136;
          *v37 = v45;
          v53 = (unsigned __int64)(v37 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *(__int64 *)((char *)v37 + v44 - 8) = v45;
          if ( (((_DWORD)v44 + (_DWORD)v37 - (_DWORD)v53) & 0xFFFFFFF8) >= 8 )
          {
            v54 = 0;
            do
            {
              v55 = v54;
              v54 += 8;
              *(_QWORD *)(v53 + v55) = v45;
            }
            while ( v54 < (((_DWORD)v44 + (_DWORD)v37 - (_DWORD)v53) & 0xFFFFFFF8) );
          }
          goto LABEL_45;
        case 15:
        case 17:
        case 18:
        case 24:
        case 25:
          v38 = 0;
          v39 = v22;
          v32 = v85;
          v33 = v22 & 0x7F;
          v40 = v22 >> 7;
          if ( !(v22 >> 7) )
            goto LABEL_32;
          break;
        case 16:
        case 19:
        case 22:
          v85[0] = v22;
          (*(void (__fastcall **)(__int64, __int64 *, __int64, __int64))(*(_QWORD *)v7 + 104LL))(v7, v85, 8, v11);
          goto LABEL_19;
        default:
          goto LABEL_142;
      }
      while ( v40 != -1 || (v39 & 0x40) == 0 )
      {
        while ( 1 )
        {
          v32 = (__int64 *)((char *)v32 + 1);
          ++v38;
          v39 = v40;
          *((_BYTE *)v32 - 1) = v33 | 0x80;
          v33 = v40 & 0x7F;
          v40 >>= 7;
          if ( v40 )
            break;
LABEL_32:
          v41 = v39 & 0x40;
          if ( !v41 )
          {
            v42 = v38 + 1;
            v37 = (__int64 *)((char *)v32 + 1);
            if ( (unsigned int)(v38 + 1) > 9 )
              goto LABEL_29;
            *(_BYTE *)v32 = v33 | 0x80;
            v43 = (char *)v32 + 1;
            goto LABEL_38;
          }
        }
      }
      v42 = v38 + 1;
      v37 = (__int64 *)((char *)v32 + 1);
      if ( (unsigned int)(v38 + 1) > 9 )
      {
LABEL_29:
        *(_BYTE *)v32 = v33;
      }
      else
      {
        LOBYTE(v41) = 127;
        *(_BYTE *)v32 = v33 | 0x80;
        v43 = (char *)v32 + 1;
LABEL_38:
        if ( v42 != 9 )
        {
          v44 = (unsigned int)(8 - v38);
          if ( (unsigned int)(v38 + 2) > 9 )
            v44 = 1;
          v45 = 0x101010101010101LL * ((unsigned __int8)v41 | 0x80u);
          if ( (unsigned int)v44 < 8 )
          {
LABEL_136:
            if ( (v44 & 4) != 0 )
            {
              *(_DWORD *)v37 = v45;
              *(_DWORD *)((char *)v37 + v44 - 4) = v45;
            }
            else if ( (_DWORD)v44 )
            {
              *(_BYTE *)v37 = v45;
              if ( (v44 & 2) != 0 )
                *(_WORD *)((char *)v37 + v44 - 2) = v45;
            }
          }
          else
          {
            *v37 = v45;
            v46 = (unsigned __int64)(v37 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *(__int64 *)((char *)v37 + v44 - 8) = v45;
            if ( (((_DWORD)v44 + (_DWORD)v37 - (_DWORD)v46) & 0xFFFFFFF8) >= 8 )
            {
              v47 = 0;
              do
              {
                v48 = v47;
                v47 += 8;
                *(_QWORD *)(v46 + v48) = v45;
              }
              while ( v47 < (((_DWORD)v44 + (_DWORD)v37 - (_DWORD)v46) & 0xFFFFFFF8) );
            }
          }
LABEL_45:
          v43 = (char *)v37 + v44;
        }
LABEL_46:
        *v43 = v41;
        LODWORD(v37) = (_DWORD)v43 + 1;
      }
LABEL_30:
      (*(void (__fastcall **)(__int64, __int64 *, _QWORD, __int64))(*(_QWORD *)v7 + 104LL))(
        v7,
        v85,
        (unsigned int)v37 - (unsigned int)v85,
        v11);
LABEL_19:
      result = i + 5;
      if ( v84 == i + 4 )
        return result;
    }
    v57 = *(_DWORD *)(a1 + 288);
    v82 = a1 + 264;
    if ( v57 )
    {
      v58 = *(_QWORD *)(a1 + 272);
      v59 = (v57 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v60 = v58 + 16LL * v59;
      v61 = *(_QWORD *)v60;
      if ( v56 == *(_QWORD *)v60 )
      {
LABEL_71:
        v22 = *(unsigned int *)(v60 + 8);
        goto LABEL_17;
      }
      v77 = 1;
      v68 = 0;
      while ( v61 != -4096 )
      {
        if ( v61 == -8192 && !v68 )
          v68 = v60;
        v59 = (v57 - 1) & (v77 + v59);
        v60 = v58 + 16LL * v59;
        v61 = *(_QWORD *)v60;
        if ( v56 == *(_QWORD *)v60 )
          goto LABEL_71;
        ++v77;
      }
      v69 = *(_DWORD *)(a1 + 280);
      if ( !v68 )
        v68 = v60;
      ++*(_QWORD *)(a1 + 264);
      v70 = v69 + 1;
      v85[0] = v68;
      if ( 4 * (v69 + 1) < 3 * v57 )
      {
        if ( v57 - *(_DWORD *)(a1 + 284) - v70 > v57 >> 3 )
        {
LABEL_96:
          *(_DWORD *)(a1 + 280) = v70;
          if ( *(_QWORD *)v68 != -4096 )
            --*(_DWORD *)(a1 + 284);
          v71 = *i;
          v22 = 0;
          *(_DWORD *)(v68 + 8) = 0;
          *(_QWORD *)v68 = v71;
          v10 = *((_DWORD *)i + 4);
          goto LABEL_17;
        }
        v72 = v57;
LABEL_101:
        sub_107DA80(v82, v72);
        sub_107C950(v82, i, v85);
        v68 = v85[0];
        v70 = *(_DWORD *)(a1 + 280) + 1;
        goto LABEL_96;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 264);
      v85[0] = 0;
    }
    v72 = 2 * v57;
    goto LABEL_101;
  }
  return result;
}
