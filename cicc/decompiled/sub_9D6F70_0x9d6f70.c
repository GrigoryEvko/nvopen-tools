// Function: sub_9D6F70
// Address: 0x9d6f70
//
__int64 *__fastcall sub_9D6F70(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v3; // rcx
  unsigned int v5; // r12d
  __int64 *v6; // rsi
  char v7; // dl
  int v8; // edx
  char v9; // al
  const char *v10; // rbx
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  _BOOL8 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int64 v23; // rdx
  unsigned int v24; // eax
  __int64 v25; // rax
  char *v26; // rsi
  unsigned __int64 v27; // rbx
  __int64 v28; // r15
  __int64 v29; // rbx
  int v30; // r15d
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int16 v34; // cx
  unsigned int v35; // ecx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  char v45; // cl
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned __int16 v48; // [rsp+20h] [rbp-360h]
  const char *v49; // [rsp+38h] [rbp-348h]
  unsigned int v50; // [rsp+38h] [rbp-348h]
  __int64 v51; // [rsp+40h] [rbp-340h]
  __int64 v52; // [rsp+40h] [rbp-340h]
  int v53; // [rsp+40h] [rbp-340h]
  __int64 v54; // [rsp+40h] [rbp-340h]
  __int64 v55; // [rsp+50h] [rbp-330h]
  int v56; // [rsp+58h] [rbp-328h]
  unsigned __int16 v57; // [rsp+5Ch] [rbp-324h]
  unsigned __int16 v58; // [rsp+5Eh] [rbp-322h]
  __int64 v61; // [rsp+70h] [rbp-310h] BYREF
  char v62; // [rsp+78h] [rbp-308h]
  __int64 v63; // [rsp+80h] [rbp-300h] BYREF
  char v64; // [rsp+88h] [rbp-2F8h]
  _BYTE *v65; // [rsp+90h] [rbp-2F0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-2E8h]
  _BYTE v67[64]; // [rsp+A0h] [rbp-2E0h] BYREF
  const char *v68; // [rsp+E0h] [rbp-2A0h] BYREF
  char *v69; // [rsp+E8h] [rbp-298h]
  __int64 v70; // [rsp+F0h] [rbp-290h]
  char v71; // [rsp+F8h] [rbp-288h] BYREF
  char v72; // [rsp+100h] [rbp-280h]
  char v73; // [rsp+101h] [rbp-27Fh]
  const char *v74; // [rsp+140h] [rbp-240h] BYREF
  __int64 v75; // [rsp+148h] [rbp-238h]
  _BYTE v76[560]; // [rsp+150h] [rbp-230h] BYREF

  v55 = a2 + 32;
  sub_A4DCE0(&v74, a2 + 32, 9, 0);
  if ( ((unsigned __int64)v74 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = (unsigned __int64)v74 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  if ( *(_QWORD *)(a2 + 1480) != *(_QWORD *)(a2 + 1488) )
  {
    v76[17] = 1;
    v74 = "Invalid multiple blocks";
    v76[16] = 3;
    sub_9C81F0(a1, a2 + 8, (__int64)&v74);
    return a1;
  }
  v5 = v2;
  v74 = v76;
  v75 = 0x4000000000LL;
  v65 = v67;
  v66 = 0x800000000LL;
  while ( 1 )
  {
    v6 = (__int64 *)v55;
    sub_9CEFB0((__int64)&v61, v55, 0, v3);
    v7 = v62 & 1;
    v62 = (2 * (v62 & 1)) | v62 & 0xFD;
    if ( v7 )
    {
      v6 = &v61;
      sub_9C9090(a1, &v61);
      goto LABEL_43;
    }
    if ( (_DWORD)v61 == 1 )
    {
      *a1 = 1;
      goto LABEL_48;
    }
    if ( (v61 & 0xFFFFFFFD) == 0 )
    {
      v73 = 1;
      v6 = (__int64 *)(a2 + 8);
      v68 = "Malformed block";
      v72 = 3;
      sub_9C81F0(a1, a2 + 8, (__int64)&v68);
      goto LABEL_43;
    }
    LODWORD(v75) = 0;
    sub_A4B600(&v63, v55, HIDWORD(v61), &v74, 0);
    v8 = v64 & 1;
    v3 = (unsigned int)(2 * v8);
    v9 = (2 * v8) | v64 & 0xFD;
    v64 = v9;
    if ( (_BYTE)v8 )
    {
      v6 = &v63;
      sub_9C8CD0(a1, &v63);
      goto LABEL_139;
    }
    if ( (_DWORD)v63 != 1 )
    {
      if ( (_DWORD)v63 != 2 )
        goto LABEL_37;
      if ( &v74[8 * (unsigned int)v75] == v74 )
      {
        v24 = v66;
      }
      else
      {
        v49 = &v74[8 * (unsigned int)v75];
        v10 = v74;
        v48 = v5;
        v11 = a2 + 1512;
        do
        {
          v12 = *(_QWORD *)v10;
          v13 = a2 + 1512;
          v14 = *(_QWORD *)(a2 + 1520);
          if ( !v14 )
            goto LABEL_22;
          do
          {
            while ( 1 )
            {
              v15 = *(_QWORD *)(v14 + 16);
              v16 = *(_QWORD *)(v14 + 24);
              if ( (unsigned int)*(_QWORD *)v10 <= *(_DWORD *)(v14 + 32) )
                break;
              v14 = *(_QWORD *)(v14 + 24);
              if ( !v16 )
                goto LABEL_20;
            }
            v13 = v14;
            v14 = *(_QWORD *)(v14 + 16);
          }
          while ( v15 );
LABEL_20:
          if ( v13 == v11 || (unsigned int)v12 < *(_DWORD *)(v13 + 32) )
          {
LABEL_22:
            v51 = v13;
            v17 = sub_22077B0(48);
            *(_DWORD *)(v17 + 32) = v12;
            v13 = v17;
            *(_QWORD *)(v17 + 40) = 0;
            v18 = sub_9D5900((_QWORD *)(a2 + 1504), v51, (unsigned int *)(v17 + 32));
            if ( v19 )
            {
              v20 = v11 == v19 || v18 || (unsigned int)v12 < *(_DWORD *)(v19 + 32);
              sub_220F040(v20, v13, v19, v11);
              ++*(_QWORD *)(a2 + 1544);
            }
            else
            {
              v52 = v18;
              j_j___libc_free_0(v13, 48);
              v13 = v52;
            }
          }
          v21 = (unsigned int)v66;
          v22 = *(_QWORD *)(v13 + 40);
          v23 = (unsigned int)v66 + 1LL;
          if ( v23 > HIDWORD(v66) )
          {
            v54 = *(_QWORD *)(v13 + 40);
            sub_C8D5F0(&v65, v67, v23, 8);
            v21 = (unsigned int)v66;
            v22 = v54;
          }
          v10 += 8;
          *(_QWORD *)&v65[8 * v21] = v22;
          v24 = v66 + 1;
          LODWORD(v66) = v66 + 1;
        }
        while ( v49 != v10 );
        v5 = v48;
      }
      v25 = sub_A7B050(*(_QWORD *)(a2 + 432), v65, v24);
      v26 = *(char **)(a2 + 1488);
      v68 = (const char *)v25;
      if ( v26 != *(char **)(a2 + 1496) )
      {
        if ( v26 )
        {
          *(_QWORD *)v26 = v25;
          v26 = *(char **)(a2 + 1488);
        }
LABEL_34:
        *(_QWORD *)(a2 + 1488) = v26 + 8;
        goto LABEL_35;
      }
      goto LABEL_135;
    }
    v56 = v75;
    if ( (v75 & 1) != 0 )
      break;
    v53 = 0;
    if ( (_DWORD)v75 )
    {
      while ( 1 )
      {
        v68 = *(const char **)(a2 + 432);
        v69 = &v71;
        v70 = 0x800000000LL;
        v27 = *(_QWORD *)&v74[8 * (v53 + 1)];
        v28 = *(_QWORD *)&v74[8 * v53];
        if ( WORD1(v27) )
        {
          _BitScanReverse64(&v44, WORD1(v27));
          v45 = 63 - (v44 ^ 0x3F);
          LODWORD(v44) = v57;
          LOBYTE(v44) = v45;
          BYTE1(v44) = 1;
          v57 = v44;
          sub_A77B90(&v68, (unsigned int)v44);
        }
        v29 = (v27 >> 11) & 0x1FFFFE00000LL | (unsigned __int16)v27;
        if ( v28 != 0xFFFFFFFFLL )
          goto LABEL_61;
        v46 = v29 & 0x400;
        if ( (v29 & 0x200) != 0 )
          break;
        if ( (v29 & 0x400) != 0 )
        {
          v47 = 85;
          goto LABEL_130;
        }
LABEL_61:
        if ( (v29 & 0x200000) != 0 )
        {
          v29 &= ~0x200000uLL;
          sub_A77CE0(&v68, 0);
        }
        if ( v29 )
        {
          v30 = 1;
          v50 = (((unsigned int)v29 & 0x1F0000) >> 16) - 1;
          while ( 1 )
          {
            v31 = (unsigned int)(v30 - 1);
            switch ( v30 )
            {
              case 4:
                v32 = 4096;
                goto LABEL_67;
              case 6:
                v32 = 0x10000000000LL;
                goto LABEL_67;
              case 16:
                v32 = 8;
                goto LABEL_67;
              case 17:
                v32 = 0x2000000;
                goto LABEL_67;
              case 19:
                v32 = 0x200000000LL;
                goto LABEL_67;
              case 21:
                v32 = (__int64)&loc_1000000;
                goto LABEL_67;
              case 22:
                v32 = 256;
                goto LABEL_67;
              case 23:
                if ( (v29 & 0x40) == 0 )
                  goto LABEL_71;
                goto LABEL_68;
              case 24:
                v32 = 0x4000000000LL;
                goto LABEL_67;
              case 28:
                v38 = 0x400000000LL;
                goto LABEL_110;
              case 31:
                v38 = 0x800000;
                goto LABEL_110;
              case 32:
                v38 = 2048;
LABEL_110:
                if ( (v38 & v29) == 0 )
                  goto LABEL_71;
                goto LABEL_68;
              case 36:
                v37 = (__int64)&dword_400000;
                goto LABEL_85;
              case 37:
                if ( (v29 & 4) == 0 )
                  goto LABEL_71;
                goto LABEL_68;
              case 42:
                v37 = 32;
                goto LABEL_85;
              case 43:
                v37 = 0x80000000LL;
                goto LABEL_85;
              case 48:
                v37 = 0x2000;
                goto LABEL_85;
              case 51:
                v37 = 512;
                goto LABEL_85;
              case 52:
                v37 = 1024;
                goto LABEL_85;
              case 53:
                v37 = 0x8000000000LL;
                goto LABEL_85;
              case 54:
                v37 = 0x20000000;
                goto LABEL_85;
              case 55:
                v37 = 2;
                goto LABEL_85;
              case 57:
                v37 = 0x100000000LL;
                goto LABEL_85;
              case 60:
                v37 = 0x2000000000LL;
                goto LABEL_85;
              case 64:
                v32 = 0x1000000000LL;
                goto LABEL_67;
              case 70:
                v32 = 0x4000;
                goto LABEL_67;
              case 71:
                v37 = 0x8000;
                goto LABEL_85;
              case 72:
                v37 = 0x800000000LL;
LABEL_85:
                if ( (v37 & v29) == 0 )
                  goto LABEL_70;
                goto LABEL_69;
              case 80:
                if ( (v29 & 1) == 0 )
                  goto LABEL_71;
                goto LABEL_68;
              case 82:
                v32 = 128;
                goto LABEL_67;
              case 86:
                if ( (v29 & 0x10) != 0 )
                  goto LABEL_81;
                goto LABEL_71;
              case 87:
                if ( (v29 & 0x1F0000) != 0 )
                {
                  v35 = v5;
                  BYTE1(v35) = 0;
                  v5 = v35;
                  if ( (unsigned int)(1LL << v50) )
                  {
                    _BitScanReverse64(&v36, (unsigned int)(1LL << v50));
                    LOBYTE(v5) = 63 - (v36 ^ 0x3F);
                    LODWORD(v36) = v5;
                    BYTE1(v36) = 1;
                    v5 = v36;
                  }
                  sub_A77B90(&v68, v5);
                }
                goto LABEL_71;
              case 95:
                if ( (v29 & 0x1C000000) != 0 )
                {
                  v33 = 1LL << ((unsigned __int8)((unsigned __int64)(v29 & 0x1C000000) >> 26) - 1);
                  v58 = (unsigned __int8)v58;
                  if ( (_DWORD)v33 )
                  {
                    _BitScanReverse64((unsigned __int64 *)&v33, (unsigned int)v33);
                    LOBYTE(v34) = 63 - (v33 ^ 0x3F);
                    HIBYTE(v34) = 1;
                    v58 = v34;
                  }
                  sub_A77BC0(&v68, v58);
                }
                goto LABEL_71;
              case 96:
                v32 = 0x40000000;
LABEL_67:
                if ( (v32 & v29) == 0 )
                  goto LABEL_71;
LABEL_68:
                if ( (unsigned int)(v30 - 81) <= 5 )
                {
LABEL_81:
                  sub_A77E60(&v68, v31, 0);
                }
                else
                {
LABEL_69:
                  sub_A77B20(&v68, v31);
LABEL_70:
                  if ( v30 == 99 )
                    goto LABEL_119;
                }
LABEL_71:
                ++v30;
                break;
              case 101:
              case 102:
                BUG();
              default:
                goto LABEL_70;
            }
          }
        }
LABEL_119:
        v39 = *(_BYTE **)&v74[8 * v53];
        v40 = sub_A7B020(*(_QWORD *)(a2 + 432), v39, &v68);
        v41 = (unsigned int)v66;
        v42 = (unsigned int)v66 + 1LL;
        if ( v42 > HIDWORD(v66) )
        {
          v39 = v67;
          sub_C8D5F0(&v65, v67, v42, 8);
          v41 = (unsigned int)v66;
        }
        *(_QWORD *)&v65[8 * v41] = v40;
        LODWORD(v66) = v66 + 1;
        if ( v69 != &v71 )
          _libc_free(v69, v39);
        v53 += 2;
        if ( v53 == v56 )
          goto LABEL_124;
      }
      BYTE1(v29) &= ~2u;
      v47 = 0;
      if ( v46 )
LABEL_130:
        BYTE1(v29) &= ~4u;
      sub_A77CD0(&v68, v47);
      goto LABEL_61;
    }
LABEL_124:
    v43 = sub_A7B050(*(_QWORD *)(a2 + 432), v65, (unsigned int)v66);
    v26 = *(char **)(a2 + 1488);
    v68 = (const char *)v43;
    if ( v26 != *(char **)(a2 + 1496) )
    {
      if ( v26 )
      {
        *(_QWORD *)v26 = v43;
        v26 = *(char **)(a2 + 1488);
      }
      goto LABEL_34;
    }
LABEL_135:
    sub_93ACB0((char **)(a2 + 1480), v26, &v68);
LABEL_35:
    v9 = v64;
    LODWORD(v66) = 0;
    if ( (v64 & 2) != 0 )
      sub_9CE230(&v63);
LABEL_37:
    if ( (v9 & 1) != 0 && v63 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v63 + 8LL))(v63);
    if ( (v62 & 2) != 0 )
LABEL_54:
      sub_9CEF10(&v61);
    if ( (v62 & 1) != 0 && v61 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 8LL))(v61);
  }
  v73 = 1;
  v6 = (__int64 *)(a2 + 8);
  v68 = "Invalid parameter attribute record";
  v72 = 3;
  sub_9C81F0(a1, a2 + 8, (__int64)&v68);
LABEL_139:
  sub_9CE2A0(&v63);
LABEL_43:
  if ( (v62 & 2) != 0 )
    goto LABEL_54;
  if ( (v62 & 1) != 0 && v61 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 8LL))(v61);
LABEL_48:
  if ( v65 != v67 )
    _libc_free(v65, v6);
  if ( v74 != v76 )
    _libc_free(v74, v6);
  return a1;
}
