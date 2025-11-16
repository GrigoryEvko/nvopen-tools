// Function: sub_AEAD90
// Address: 0xaead90
//
__int64 __fastcall sub_AEAD90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r11
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r10
  __int64 v13; // rbx
  bool v14; // zf
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rsi
  int v21; // eax
  unsigned __int8 v22; // al
  __int64 v23; // rdi
  __int64 (__fastcall *v24)(__int64, __int64, __int64, __int64); // rsi
  __int64 v25; // rax
  unsigned __int8 **v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rbx
  unsigned __int8 **v29; // r15
  unsigned __int8 **v30; // r14
  __int64 v31; // rax
  unsigned int v32; // eax
  unsigned __int8 v33; // al
  __int64 v34; // rdi
  __int64 v35; // rax
  unsigned __int8 **v36; // rdx
  unsigned __int8 **v37; // r8
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int8 **v41; // r15
  unsigned __int8 **v42; // rbx
  unsigned __int8 **v43; // r8
  char v44; // al
  unsigned int v45; // edi
  __int64 *v46; // rax
  __int64 v47; // rcx
  __int64 *v48; // rax
  unsigned int v49; // ecx
  int v50; // eax
  __int64 *v51; // rdx
  __int64 v52; // rdi
  int v53; // r11d
  __int64 *v54; // r8
  int v55; // r11d
  unsigned __int8 **v56; // r8
  unsigned __int8 **v57; // r8
  unsigned __int8 **v58; // r8
  __int64 *v59; // rdi
  int v60; // r11d
  unsigned int v61; // r12d
  int v62; // r8d
  char v63; // al
  char v64; // al
  char v65; // al
  __int64 v66; // [rsp+8h] [rbp-1C8h]
  __int64 v67; // [rsp+10h] [rbp-1C0h]
  __int64 v68; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 **v69; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 **v70; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 **v71; // [rsp+18h] [rbp-1B8h]
  __int64 v72; // [rsp+20h] [rbp-1B0h]
  unsigned __int8 **v73; // [rsp+20h] [rbp-1B0h]
  __int64 v74; // [rsp+20h] [rbp-1B0h]
  __int64 v75; // [rsp+20h] [rbp-1B0h]
  __int64 v76; // [rsp+20h] [rbp-1B0h]
  __int64 v77; // [rsp+20h] [rbp-1B0h]
  __int64 v78; // [rsp+20h] [rbp-1B0h]
  __int64 v79; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v80; // [rsp+37h] [rbp-199h]
  __int64 v81; // [rsp+38h] [rbp-198h]
  __int64 v82; // [rsp+40h] [rbp-190h]
  __int64 v83; // [rsp+48h] [rbp-188h]
  _QWORD v84[2]; // [rsp+50h] [rbp-180h] BYREF
  __int64 v85; // [rsp+60h] [rbp-170h] BYREF
  __int64 v86; // [rsp+68h] [rbp-168h]
  __int64 v87; // [rsp+70h] [rbp-160h]
  unsigned int v88; // [rsp+78h] [rbp-158h]
  __int64 v89; // [rsp+80h] [rbp-150h] BYREF
  void *s; // [rsp+88h] [rbp-148h]
  _BYTE v91[12]; // [rsp+90h] [rbp-140h]
  char v92; // [rsp+9Ch] [rbp-134h]
  __int64 v93; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v94; // [rsp+E0h] [rbp-F0h] BYREF
  char *v95; // [rsp+E8h] [rbp-E8h]
  __int64 v96; // [rsp+F0h] [rbp-E0h]
  int v97; // [rsp+F8h] [rbp-D8h]
  char v98; // [rsp+FCh] [rbp-D4h]
  char v99; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v100; // [rsp+140h] [rbp-90h] BYREF
  char *v101; // [rsp+148h] [rbp-88h]
  __int64 v102; // [rsp+150h] [rbp-80h]
  int v103; // [rsp+158h] [rbp-78h]
  char v104; // [rsp+15Ch] [rbp-74h]
  char v105; // [rsp+160h] [rbp-70h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 && (a2 = 0, sub_B91C10(a1, 0)) )
  {
    a2 = 0;
    sub_B994C0(a1, 0);
    v80 = 1;
  }
  else
  {
    v80 = 0;
  }
  v3 = *(_QWORD *)(a1 + 80);
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v79 = v3;
  v66 = a1 + 72;
  if ( v3 == a1 + 72 )
  {
    v18 = 0;
    v19 = 0;
    goto LABEL_32;
  }
  do
  {
    if ( !v79 )
      BUG();
    v81 = v79 + 24;
    v83 = *(_QWORD *)(v79 + 32);
    if ( v79 + 24 != v83 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v13 = v83;
          v14 = *(_BYTE *)(v83 - 24) == 85;
          v15 = v83 - 24;
          v83 = *(_QWORD *)(v83 + 8);
          if ( !v14 )
            break;
          v16 = *(_QWORD *)(v13 - 56);
          if ( !v16 )
            break;
          if ( *(_BYTE *)v16 )
            break;
          v17 = *(_QWORD *)(v13 + 56);
          if ( *(_QWORD *)(v16 + 24) != v17
            || (*(_BYTE *)(v16 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v16 + 36) - 68) > 3 )
          {
            break;
          }
          sub_B43D60(v15, a2, a3, v17);
          v80 = 1;
          if ( v81 == v83 )
            goto LABEL_30;
        }
        a2 = *(_QWORD *)(v13 + 24);
        if ( a2 )
        {
          v100 = 0;
          if ( (__int64 *)(v13 + 24) != &v100 )
          {
            sub_B91220(v13 + 24);
            a2 = v100;
            *(_QWORD *)(v13 + 24) = v100;
            if ( a2 )
              sub_B976B0(&v100, a2, v13 + 24, v4, v5, v6);
          }
          v80 = 1;
        }
        if ( (*(_BYTE *)(v13 - 17) & 0x20) != 0 )
          break;
LABEL_22:
        sub_B44570(v15);
        if ( v81 == v83 )
          goto LABEL_30;
      }
      a2 = 18;
      v7 = sub_B91C10(v15, 18);
      v82 = v7;
      v8 = v7;
      if ( !v7 )
      {
LABEL_20:
        if ( (*(_BYTE *)(v13 - 17) & 0x20) != 0 )
        {
          sub_B9A090(v15, "heapallocsite", 13, 0);
          a2 = 38;
          sub_B99FD0(v15, 38, 0);
        }
        goto LABEL_22;
      }
      a2 = v86;
      if ( v88 )
      {
        v9 = (v88 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v10 = (__int64 *)(v86 + 16LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
        {
LABEL_17:
          v12 = v10[1];
          if ( v12 )
            goto LABEL_18;
        }
        else
        {
          v21 = 1;
          while ( v11 != -4096 )
          {
            v62 = v21 + 1;
            v9 = (v88 - 1) & (v21 + v9);
            v10 = (__int64 *)(v86 + 16LL * v9);
            v11 = *v10;
            if ( v82 == *v10 )
              goto LABEL_17;
            v21 = v62;
          }
        }
      }
      v104 = 1;
      s = &v93;
      v95 = &v99;
      v101 = &v105;
      *(_QWORD *)v91 = 0x100000008LL;
      v94 = 0;
      v98 = 1;
      v96 = 8;
      v97 = 0;
      v100 = 0;
      v102 = 8;
      v103 = 0;
      *(_DWORD *)&v91[8] = 0;
      v92 = 1;
      v93 = v82;
      v89 = 1;
      v72 = v82 - 16;
      v22 = *(_BYTE *)(v82 - 16);
      if ( (v22 & 2) != 0 )
      {
        v23 = *(_QWORD *)(v82 - 32);
        v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))*(unsigned int *)(v82 - 24);
      }
      else
      {
        v23 = v72 - 8LL * ((v22 >> 2) & 0xF);
        v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))((*(_WORD *)(v82 - 16) >> 6) & 0xF);
      }
      v25 = sub_AE5C70(v23, (__int64)v24, 1);
      if ( (unsigned __int8 **)v25 == v26 )
      {
        v12 = v82;
        goto LABEL_63;
      }
      v27 = 0;
      v68 = v13;
      v67 = v15;
      v28 = 0;
      v29 = (unsigned __int8 **)v25;
      v30 = v26;
      do
      {
        v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v94;
        v28 -= !sub_AE70C0((__int64)&v89, (__int64)&v94, *v29++, v27) - 1LL;
      }
      while ( v30 != v29 );
      v31 = v28;
      v15 = v67;
      v13 = v68;
      v12 = v82;
      if ( !v31 )
        goto LABEL_59;
      ++v89;
      if ( !v92 )
      {
        v32 = 4 * (*(_DWORD *)&v91[4] - *(_DWORD *)&v91[8]);
        if ( v32 < 0x20 )
          v32 = 32;
        if ( v32 < *(_DWORD *)v91 )
        {
          sub_C8C990(&v89);
LABEL_47:
          v33 = *(_BYTE *)(v82 - 16);
          if ( (v33 & 2) != 0 )
          {
            v34 = *(_QWORD *)(v82 - 32);
            v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))*(unsigned int *)(v82 - 24);
          }
          else
          {
            v34 = v72 - 8LL * ((v33 >> 2) & 0xF);
            v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))((*(_WORD *)(v82 - 16) >> 6) & 0xF);
          }
          v35 = sub_AE5C70(v34, (__int64)v24, 1);
          v73 = v36;
          v37 = (unsigned __int8 **)v35;
          v38 = (__int64)v36 - v35;
          v39 = v38 >> 5;
          v40 = v38 >> 3;
          if ( v39 > 0 )
          {
            v41 = v37;
            v42 = &v37[4 * v39];
            while ( 1 )
            {
              v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
              if ( !(unsigned __int8)sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, *v41) )
              {
                v43 = v41;
                v13 = v68;
                v15 = v67;
                goto LABEL_57;
              }
              v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
              if ( !(unsigned __int8)sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, v41[1]) )
              {
                v56 = v41;
                v13 = v68;
                v15 = v67;
                v43 = v56 + 1;
                goto LABEL_57;
              }
              v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
              if ( !(unsigned __int8)sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, v41[2]) )
              {
                v57 = v41;
                v13 = v68;
                v15 = v67;
                v43 = v57 + 2;
                goto LABEL_57;
              }
              v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
              if ( !(unsigned __int8)sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, v41[3]) )
                break;
              v41 += 4;
              if ( v41 == v42 )
              {
                v37 = v41;
                v13 = v68;
                v15 = v67;
                v40 = v73 - v37;
                goto LABEL_90;
              }
            }
            v58 = v41;
            v13 = v68;
            v15 = v67;
            v43 = v58 + 3;
LABEL_57:
            if ( v73 != v43 )
            {
              v24 = sub_AE6EB0;
              v84[0] = &v100;
              v84[1] = &v94;
              v12 = sub_AE5D10(v82, (__int64 (__fastcall *)(__int64))sub_AE6EB0, (__int64)v84);
              goto LABEL_59;
            }
            goto LABEL_93;
          }
LABEL_90:
          if ( v40 != 2 )
          {
            if ( v40 != 3 )
            {
              if ( v40 != 1 )
              {
LABEL_93:
                v12 = 0;
                goto LABEL_59;
              }
              goto LABEL_110;
            }
            v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
            v69 = v37;
            v63 = sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, *v37);
            v43 = v69;
            if ( !v63 )
              goto LABEL_57;
            v37 = v69 + 1;
          }
          v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
          v70 = v37;
          v64 = sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, *v37);
          v43 = v70;
          if ( !v64 )
            goto LABEL_57;
          v37 = v70 + 1;
LABEL_110:
          v24 = (__int64 (__fastcall *)(__int64, __int64, __int64, __int64))&v100;
          v71 = v37;
          v65 = sub_AE7280((__int64)&v89, (__int64)&v100, (__int64)&v94, *v37);
          v12 = 0;
          v43 = v71;
          if ( !v65 )
            goto LABEL_57;
LABEL_59:
          if ( v104 )
          {
            v44 = v98;
          }
          else
          {
            v75 = v12;
            _libc_free(v101, v24);
            v44 = v98;
            v12 = v75;
          }
          if ( !v44 )
          {
            v74 = v12;
            _libc_free(v95, v24);
            v12 = v74;
          }
LABEL_63:
          if ( v92 )
          {
            a2 = v88;
            if ( v88 )
              goto LABEL_65;
          }
          else
          {
            v76 = v12;
            _libc_free(s, v24);
            a2 = v88;
            v12 = v76;
            if ( v88 )
            {
LABEL_65:
              v45 = (a2 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
              v46 = (__int64 *)(v86 + 16LL * v45);
              v47 = *v46;
              if ( v82 == *v46 )
              {
LABEL_66:
                v48 = v46 + 1;
LABEL_67:
                *v48 = v12;
LABEL_18:
                if ( v12 != v82 )
                {
                  a2 = 18;
                  sub_B99FD0(v15, 18, v12);
                }
                goto LABEL_20;
              }
              v55 = 1;
              v51 = 0;
              while ( v47 != -4096 )
              {
                if ( !v51 && v47 == -8192 )
                  v51 = v46;
                v45 = (a2 - 1) & (v55 + v45);
                v46 = (__int64 *)(v86 + 16LL * v45);
                v47 = *v46;
                if ( v82 == *v46 )
                  goto LABEL_66;
                ++v55;
              }
              if ( !v51 )
                v51 = v46;
              ++v85;
              v50 = v87 + 1;
              if ( 4 * ((int)v87 + 1) < (unsigned int)(3 * a2) )
              {
                if ( (int)a2 - HIDWORD(v87) - v50 <= (unsigned int)a2 >> 3 )
                {
                  v78 = v12;
                  sub_AEABB0((__int64)&v85, a2);
                  if ( !v88 )
                  {
LABEL_131:
                    LODWORD(v87) = v87 + 1;
                    BUG();
                  }
                  v59 = 0;
                  v60 = 1;
                  v61 = (v88 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
                  v12 = v78;
                  v50 = v87 + 1;
                  v51 = (__int64 *)(v86 + 16LL * v61);
                  a2 = *v51;
                  if ( v82 != *v51 )
                  {
                    while ( a2 != -4096 )
                    {
                      if ( a2 == -8192 && !v59 )
                        v59 = v51;
                      v61 = (v88 - 1) & (v60 + v61);
                      v51 = (__int64 *)(v86 + 16LL * v61);
                      a2 = *v51;
                      if ( v82 == *v51 )
                        goto LABEL_86;
                      ++v60;
                    }
                    if ( v59 )
                      v51 = v59;
                  }
                }
                goto LABEL_86;
              }
LABEL_71:
              v77 = v12;
              sub_AEABB0((__int64)&v85, 2 * a2);
              if ( !v88 )
                goto LABEL_131;
              a2 = v88 - 1;
              v12 = v77;
              v49 = a2 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
              v50 = v87 + 1;
              v51 = (__int64 *)(v86 + 16LL * v49);
              v52 = *v51;
              if ( v82 != *v51 )
              {
                v53 = 1;
                v54 = 0;
                while ( v52 != -4096 )
                {
                  if ( !v54 && v52 == -8192 )
                    v54 = v51;
                  v49 = a2 & (v53 + v49);
                  v51 = (__int64 *)(v86 + 16LL * v49);
                  v52 = *v51;
                  if ( v82 == *v51 )
                    goto LABEL_86;
                  ++v53;
                }
                if ( v54 )
                  v51 = v54;
              }
LABEL_86:
              LODWORD(v87) = v50;
              if ( *v51 != -4096 )
                --HIDWORD(v87);
              v51[1] = 0;
              *v51 = v82;
              v48 = v51 + 1;
              goto LABEL_67;
            }
          }
          ++v85;
          goto LABEL_71;
        }
        memset(s, -1, 8LL * *(unsigned int *)v91);
      }
      *(_QWORD *)&v91[4] = 0;
      goto LABEL_47;
    }
LABEL_30:
    v79 = *(_QWORD *)(v79 + 8);
  }
  while ( v66 != v79 );
  v18 = v86;
  v19 = 16LL * v88;
LABEL_32:
  sub_C7D6A0(v18, v19, 8);
  return v80;
}
