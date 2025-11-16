// Function: sub_324FBC0
// Address: 0x324fbc0
//
void __fastcall sub_324FBC0(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  char v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdx
  const void *v13; // rcx
  size_t v14; // rdx
  size_t v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r13
  unsigned __int8 v25; // al
  int v26; // eax
  char v27; // r8
  unsigned __int8 v28; // al
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 (*v31)(void); // rax
  int v32; // edx
  int v33; // eax
  unsigned __int8 v34; // al
  __int64 v35; // rdi
  __int64 v36; // rdx
  unsigned __int8 v37; // al
  __int64 v38; // rdi
  const void *v39; // rcx
  size_t v40; // rdx
  size_t v41; // r8
  __int64 *v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 **v45; // r14
  unsigned __int8 v46; // al
  unsigned __int64 v47; // r14
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // r8
  __int64 v51; // rdi
  unsigned __int64 *v52; // rax
  unsigned __int64 v53; // rcx
  int v54; // edx
  unsigned int v55; // eax
  unsigned __int64 *v56; // rdx
  int v57; // eax
  int v58; // eax
  int v59; // esi
  int v60; // esi
  __int64 v61; // r8
  unsigned int v62; // ecx
  unsigned __int64 v63; // rdi
  int v64; // r11d
  unsigned __int64 *v65; // r10
  int v66; // esi
  int v67; // esi
  int v68; // r11d
  __int64 v69; // r8
  unsigned int v70; // ecx
  unsigned __int64 v71; // rdi
  int v72; // [rsp+14h] [rbp-4Ch]
  unsigned int v73; // [rsp+14h] [rbp-4Ch]
  __int64 v74; // [rsp+18h] [rbp-48h]

  if ( !a4 || *(_BYTE *)(*(_QWORD *)(a1 + 80) + 42LL) )
  {
    v8 = sub_324D2E0((__int64 *)a1, a2, a3, a4);
    if ( v8 )
      return;
    v74 = a2 - 16;
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) == 0 )
    {
LABEL_4:
      v10 = *(_QWORD *)(v74 - 8LL * ((v9 >> 2) & 0xF) + 16);
      if ( !v10 )
        goto LABEL_75;
      goto LABEL_5;
    }
  }
  else
  {
    v8 = a4;
    v74 = a2 - 16;
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) == 0 )
      goto LABEL_4;
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
  if ( !v10 )
  {
LABEL_12:
    v16 = 0;
    if ( *(_DWORD *)(a2 - 24) <= 0xBu )
      goto LABEL_15;
    v17 = *(_QWORD *)(a2 - 32);
    goto LABEL_14;
  }
LABEL_5:
  sub_B91420(v10);
  v9 = *(_BYTE *)(a2 - 16);
  if ( v11 )
  {
    if ( (v9 & 2) != 0 )
      v12 = *(_QWORD *)(a2 - 32);
    else
      v12 = v74 - 8LL * ((v9 >> 2) & 0xF);
    v13 = *(const void **)(v12 + 16);
    if ( v13 )
    {
      v13 = (const void *)sub_B91420(*(_QWORD *)(v12 + 16));
      v15 = v14;
    }
    else
    {
      v15 = 0;
    }
    sub_324AD70((__int64 *)a1, a3, 3, v13, v15);
    v9 = *(_BYTE *)(a2 - 16);
  }
  if ( (v9 & 2) != 0 )
    goto LABEL_12;
LABEL_75:
  if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) > 0xB )
  {
    v17 = v74 - 8LL * ((v9 >> 2) & 0xF);
LABEL_14:
    v16 = *(_QWORD *)(v17 + 88);
    goto LABEL_15;
  }
  v16 = 0;
LABEL_15:
  sub_324CC60((__int64 *)a1, a3, v16);
  if ( !v8 )
    sub_3249D90((__int64 *)a1, a3, a2);
  if ( !a4 )
  {
    if ( (*(_BYTE *)(a2 + 33) & 1) == 0 )
      goto LABEL_19;
    v54 = *(_DWORD *)(*(_QWORD *)(a1 + 80) + 16LL);
    if ( (unsigned __int16)v54 <= 0x42u )
    {
      if ( (_WORD)v54 )
      {
        switch ( (__int16)v54 )
        {
          case 1:
          case 2:
          case 12:
          case 16:
          case 29:
          case 44:
            sub_3249FA0((__int64 *)a1, a3, 39);
            goto LABEL_19;
          case 3:
          case 4:
          case 5:
          case 6:
          case 7:
          case 8:
          case 9:
          case 10:
          case 11:
          case 13:
          case 14:
          case 15:
          case 17:
          case 18:
          case 19:
          case 20:
          case 21:
          case 22:
          case 23:
          case 24:
          case 25:
          case 26:
          case 27:
          case 28:
          case 30:
          case 31:
          case 32:
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 42:
          case 43:
          case 45:
          case 46:
          case 47:
          case 48:
          case 49:
          case 50:
          case 51:
          case 52:
          case 53:
          case 54:
          case 55:
          case 56:
          case 57:
          case 61:
          case 64:
          case 65:
          case 66:
            goto LABEL_19;
          default:
            goto LABEL_110;
        }
      }
LABEL_165:
      BUG();
    }
    if ( (unsigned __int16)v54 != 45056 )
    {
      if ( (unsigned __int16)v54 <= 0xB000u )
      {
        if ( (unsigned __int16)v54 > 0x8001u || (v54 & 0x8000) != 0 )
          goto LABEL_19;
      }
      else if ( (unsigned __int16)v54 == 0xFFFF )
      {
        goto LABEL_19;
      }
LABEL_110:
      if ( (unsigned int)(unsigned __int16)v54 - 32769 > 0x7FFD )
        goto LABEL_165;
    }
LABEL_19:
    if ( (*(_BYTE *)(a2 + 37) & 8) != 0 )
      sub_3249FA0((__int64 *)a1, a3, 16366);
    v18 = *(_BYTE *)(a2 - 16);
    if ( (v18 & 2) != 0 )
      v19 = *(_QWORD *)(a2 - 32);
    else
      v19 = v74 - 8LL * ((v18 >> 2) & 0xF);
    v20 = *(_QWORD *)(v19 + 32);
    if ( v20 )
    {
      v21 = *(_BYTE *)(v20 - 16);
      if ( (v21 & 2) != 0 )
        v22 = *(_QWORD *)(v20 - 32);
      else
        v22 = v20 - 16 - 8LL * ((v21 >> 2) & 0xF);
      v23 = *(unsigned __int8 *)(v20 + 44);
      v24 = *(_QWORD *)(v22 + 24);
      if ( (unsigned __int8)v23 > 1u )
        sub_3249A20((__int64 *)a1, (unsigned __int64 **)(a3 + 8), 54, 65547, v23);
      if ( v24 )
      {
        v25 = *(_BYTE *)(v24 - 16);
        if ( (v25 & 2) != 0 )
        {
          if ( !*(_DWORD *)(v24 - 24) )
            goto LABEL_31;
          v42 = *(__int64 **)(v24 - 32);
        }
        else
        {
          if ( (*(_WORD *)(v24 - 16) & 0x3C0) == 0 )
            goto LABEL_31;
          v42 = (__int64 *)(v24 - 16 - 8LL * ((v25 >> 2) & 0xF));
        }
        v43 = *v42;
        if ( v43 )
        {
          sub_32495E0((__int64 *)a1, a3, v43, 73);
          v26 = *(_DWORD *)(a2 + 36);
          v27 = v26 & 3;
          if ( (v26 & 3) == 0 )
          {
LABEL_32:
            if ( (v26 & 8) != 0 )
              goto LABEL_33;
            goto LABEL_93;
          }
LABEL_82:
          sub_3249A20((__int64 *)a1, (unsigned __int64 **)(a3 + 8), 76, 65547, v27 & 3);
          if ( *(_DWORD *)(a2 + 24) != -1 )
          {
            v44 = sub_A777F0(0x10u, (__int64 *)(a1 + 88));
            v45 = (unsigned __int64 **)v44;
            if ( v44 )
            {
              *(_QWORD *)v44 = 0;
              *(_DWORD *)(v44 + 8) = 0;
            }
            sub_3249B00((__int64 *)a1, (unsigned __int64 **)v44, 11, 16);
            sub_3249B00((__int64 *)a1, v45, 15, *(unsigned int *)(a2 + 24));
            sub_3249620((__int64 *)a1, a3, 77, (__int64 **)v45);
          }
          v46 = *(_BYTE *)(a2 - 16);
          if ( (v46 & 2) != 0 )
          {
            v47 = 0;
            if ( *(_DWORD *)(a2 - 24) <= 8u )
              goto LABEL_90;
            v48 = *(_QWORD *)(a2 - 32);
          }
          else
          {
            if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) <= 8 )
            {
              v47 = 0;
LABEL_90:
              v49 = *(_DWORD *)(a1 + 336);
              if ( v49 )
              {
                v50 = *(_QWORD *)(a1 + 320);
                v51 = (v49 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
                v52 = (unsigned __int64 *)(v50 + 16 * v51);
                v53 = *v52;
                if ( a3 == *v52 )
                  goto LABEL_92;
                v72 = 1;
                v56 = 0;
                while ( v53 != -4096 )
                {
                  if ( !v56 && v53 == -8192 )
                    v56 = v52;
                  LODWORD(v51) = (v49 - 1) & (v72 + v51);
                  v52 = (unsigned __int64 *)(v50 + 16LL * (unsigned int)v51);
                  v53 = *v52;
                  if ( a3 == *v52 )
                    goto LABEL_92;
                  ++v72;
                }
                if ( !v56 )
                  v56 = v52;
                v57 = *(_DWORD *)(a1 + 328);
                ++*(_QWORD *)(a1 + 312);
                v58 = v57 + 1;
                if ( 4 * v58 < 3 * v49 )
                {
                  if ( v49 - *(_DWORD *)(a1 + 332) - v58 > v49 >> 3 )
                  {
LABEL_132:
                    *(_DWORD *)(a1 + 328) = v58;
                    if ( *v56 != -4096 )
                      --*(_DWORD *)(a1 + 332);
                    *v56 = a3;
                    v56[1] = v47;
LABEL_92:
                    if ( (*(_DWORD *)(a2 + 36) & 8) != 0 )
                    {
LABEL_33:
                      v28 = *(_BYTE *)(a2 - 16);
                      if ( (v28 & 2) != 0 )
                      {
LABEL_34:
                        v29 = 0;
                        if ( *(_DWORD *)(a2 - 24) <= 0xAu )
                          goto LABEL_37;
                        v30 = *(_QWORD *)(a2 - 32);
                        goto LABEL_36;
                      }
LABEL_94:
                      if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) <= 0xA )
                      {
                        v29 = 0;
LABEL_37:
                        sub_324C7E0((__int64 *)a1, a3, v29);
                        if ( (*(_BYTE *)(a2 + 32) & 0x40) != 0 )
                          sub_3249FA0((__int64 *)a1, a3, 52);
                        if ( (*(_BYTE *)(a2 + 36) & 4) == 0 )
                          sub_3249FA0((__int64 *)a1, a3, 63);
                        if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 3768LL) )
                        {
                          if ( (*(_BYTE *)(a2 + 36) & 0x10) != 0 )
                            sub_3249FA0((__int64 *)a1, a3, 16353);
                          v31 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 184) + 448LL);
                          if ( v31 != sub_3020040 )
                          {
                            v55 = v31();
                            if ( v55 )
                              sub_3249A20((__int64 *)a1, (unsigned __int64 **)(a3 + 8), 16355, 65548, v55);
                          }
                        }
                        v32 = *(_DWORD *)(a2 + 32);
                        if ( (v32 & 0x2000) != 0 )
                        {
                          sub_3249FA0((__int64 *)a1, a3, 119);
                          v32 = *(_DWORD *)(a2 + 32);
                        }
                        if ( (v32 & 0x4000) != 0 )
                        {
                          sub_3249FA0((__int64 *)a1, a3, 120);
                          v32 = *(_DWORD *)(a2 + 32);
                          if ( (v32 & 0x100000) == 0 )
                            goto LABEL_49;
                        }
                        else if ( (v32 & 0x100000) == 0 )
                        {
                          goto LABEL_49;
                        }
                        sub_3249FA0((__int64 *)a1, a3, 135);
                        v32 = *(_DWORD *)(a2 + 32);
LABEL_49:
                        sub_3249F00((__int64 *)a1, a3, v32);
                        if ( *(char *)(a2 + 32) < 0 )
                          sub_3249FA0((__int64 *)a1, a3, 99);
                        v33 = *(_DWORD *)(a2 + 36);
                        if ( (v33 & 0x100) != 0 )
                        {
                          sub_3249FA0((__int64 *)a1, a3, 106);
                          v33 = *(_DWORD *)(a2 + 36);
                        }
                        if ( (v33 & 0x20) != 0 )
                        {
                          sub_3249FA0((__int64 *)a1, a3, 103);
                          v33 = *(_DWORD *)(a2 + 36);
                        }
                        if ( (v33 & 0x40) != 0 )
                        {
                          sub_3249FA0((__int64 *)a1, a3, 102);
                          v33 = *(_DWORD *)(a2 + 36);
                        }
                        if ( (v33 & 0x80u) != 0 )
                          sub_3249FA0((__int64 *)a1, a3, 104);
                        v34 = *(_BYTE *)(a2 - 16);
                        if ( (v34 & 2) != 0 )
                        {
                          if ( *(_DWORD *)(a2 - 24) > 0xCu )
                          {
                            v35 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 96LL);
                            if ( v35 )
                              goto LABEL_62;
                          }
                        }
                        else if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) > 0xC )
                        {
                          v35 = *(_QWORD *)(v74 - 8LL * ((v34 >> 2) & 0xF) + 96);
                          if ( v35 )
                          {
LABEL_62:
                            sub_B91420(v35);
                            if ( !v36 )
                              goto LABEL_68;
                            v37 = *(_BYTE *)(a2 - 16);
                            if ( (v37 & 2) != 0 )
                            {
                              if ( *(_DWORD *)(a2 - 24) > 0xCu )
                              {
                                v38 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 96LL);
                                if ( v38 )
                                {
LABEL_66:
                                  v39 = (const void *)sub_B91420(v38);
                                  v41 = v40;
LABEL_67:
                                  sub_324AD70((__int64 *)a1, a3, 86, v39, v41);
                                  goto LABEL_68;
                                }
                              }
                            }
                            else if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) > 0xC )
                            {
                              v38 = *(_QWORD *)(v74 - 8LL * ((v37 >> 2) & 0xF) + 96);
                              if ( v38 )
                                goto LABEL_66;
                            }
                            v41 = 0;
                            v39 = 0;
                            goto LABEL_67;
                          }
                        }
LABEL_68:
                        if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 4u && (*(_BYTE *)(a2 + 37) & 2) != 0 )
                          sub_3249FA0((__int64 *)a1, a3, 138);
                        return;
                      }
                      v30 = v74 - 8LL * ((v28 >> 2) & 0xF);
LABEL_36:
                      v29 = *(_QWORD *)(v30 + 80);
                      goto LABEL_37;
                    }
LABEL_93:
                    sub_3249FA0((__int64 *)a1, a3, 60);
                    sub_324C890((__int64 *)a1, a3, v24);
                    v28 = *(_BYTE *)(a2 - 16);
                    if ( (v28 & 2) != 0 )
                      goto LABEL_34;
                    goto LABEL_94;
                  }
                  v73 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
                  sub_324F9E0(a1 + 312, v49);
                  v66 = *(_DWORD *)(a1 + 336);
                  if ( v66 )
                  {
                    v67 = v66 - 1;
                    v68 = 1;
                    v65 = 0;
                    v69 = *(_QWORD *)(a1 + 320);
                    v70 = v67 & v73;
                    v58 = *(_DWORD *)(a1 + 328) + 1;
                    v56 = (unsigned __int64 *)(v69 + 16LL * (v67 & v73));
                    v71 = *v56;
                    if ( a3 == *v56 )
                      goto LABEL_132;
                    while ( v71 != -4096 )
                    {
                      if ( v71 == -8192 && !v65 )
                        v65 = v56;
                      v70 = v67 & (v68 + v70);
                      v56 = (unsigned __int64 *)(v69 + 16LL * v70);
                      v71 = *v56;
                      if ( a3 == *v56 )
                        goto LABEL_132;
                      ++v68;
                    }
                    goto LABEL_140;
                  }
LABEL_164:
                  ++*(_DWORD *)(a1 + 328);
                  goto LABEL_165;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 312);
              }
              sub_324F9E0(a1 + 312, 2 * v49);
              v59 = *(_DWORD *)(a1 + 336);
              if ( v59 )
              {
                v60 = v59 - 1;
                v61 = *(_QWORD *)(a1 + 320);
                v62 = v60 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
                v58 = *(_DWORD *)(a1 + 328) + 1;
                v56 = (unsigned __int64 *)(v61 + 16LL * v62);
                v63 = *v56;
                if ( a3 == *v56 )
                  goto LABEL_132;
                v64 = 1;
                v65 = 0;
                while ( v63 != -4096 )
                {
                  if ( !v65 && v63 == -8192 )
                    v65 = v56;
                  v62 = v60 & (v64 + v62);
                  v56 = (unsigned __int64 *)(v61 + 16LL * v62);
                  v63 = *v56;
                  if ( a3 == *v56 )
                    goto LABEL_132;
                  ++v64;
                }
LABEL_140:
                if ( v65 )
                  v56 = v65;
                goto LABEL_132;
              }
              goto LABEL_164;
            }
            v48 = v74 - 8LL * ((v46 >> 2) & 0xF);
          }
          v47 = *(_QWORD *)(v48 + 64);
          goto LABEL_90;
        }
      }
    }
    else
    {
      v24 = 0;
    }
LABEL_31:
    v26 = *(_DWORD *)(a2 + 36);
    v27 = v26 & 3;
    if ( (v26 & 3) == 0 )
      goto LABEL_32;
    goto LABEL_82;
  }
}
