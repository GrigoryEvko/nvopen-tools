// Function: sub_2CFD5A0
// Address: 0x2cfd5a0
//
__int64 __fastcall sub_2CFD5A0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  char v8; // al
  __int64 v9; // r15
  size_t v10; // rdx
  char *v11; // rsi
  __int64 v12; // r8
  _BYTE *v13; // r10
  char v14; // al
  size_t v15; // rdx
  const char *v16; // r15
  size_t v17; // rax
  const char *v18; // rsi
  unsigned int v19; // r15d
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rax
  _QWORD *v23; // rdx
  _QWORD *v24; // rdx
  _QWORD *v25; // rdx
  _QWORD *v26; // rdx
  _QWORD *v27; // rdx
  __int64 v28; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  char *v33; // rax
  unsigned __int8 v34; // dl
  __int64 v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // r10
  char *v38; // rsi
  size_t v39; // rax
  size_t v40; // rdx
  __int64 v41; // rax
  size_t v42; // rdx
  char *v43; // rsi
  size_t v44; // rax
  __int64 v45; // r15
  char v46; // al
  __int64 v47; // r10
  char v48; // dl
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rdi
  char v52; // al
  char v53; // al
  char v54; // al
  __int64 v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned int v61; // esi
  __int64 *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // [rsp+0h] [rbp-50h]
  __int64 v67; // [rsp+8h] [rbp-48h]
  __int64 v68; // [rsp+8h] [rbp-48h]
  __int64 v69; // [rsp+10h] [rbp-40h]
  __int64 v70; // [rsp+10h] [rbp-40h]
  __int64 v71; // [rsp+10h] [rbp-40h]
  __int64 v72; // [rsp+10h] [rbp-40h]
  __int64 v73; // [rsp+10h] [rbp-40h]
  __int64 v74; // [rsp+10h] [rbp-40h]
  __int64 v75; // [rsp+18h] [rbp-38h]
  __int64 v76; // [rsp+18h] [rbp-38h]
  char *v77; // [rsp+18h] [rbp-38h]
  __int64 v78; // [rsp+18h] [rbp-38h]
  char *v79; // [rsp+18h] [rbp-38h]
  __int64 v80; // [rsp+18h] [rbp-38h]
  __int64 v81; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_OWORD *)(a1 + 40) = 0;
  v8 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 62 )
  {
    v9 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v9 <= 0x1Cu )
      return 0;
    v10 = 0;
    v11 = off_4C5D0D8[0];
    if ( off_4C5D0D8[0] )
    {
      v11 = off_4C5D0D8[0];
      v10 = strlen(off_4C5D0D8[0]);
    }
    if ( !*(_QWORD *)(v9 + 48) && (*(_BYTE *)(v9 + 7) & 0x20) == 0 )
      return 0;
    v12 = sub_B91F50(v9, v11, v10);
    if ( v12 )
    {
      v13 = *(_BYTE **)(a2 - 64);
      v14 = *v13;
      if ( *v13 <= 0x1Cu )
      {
        v76 = *(_QWORD *)(a2 - 64);
        if ( v14 != 22 )
        {
LABEL_9:
          v15 = 0;
          v16 = off_4C5D0D8[0];
          if ( !off_4C5D0D8[0] )
          {
LABEL_11:
            v18 = v16;
            v19 = 1;
            sub_B9A090(a2, v18, v15, v12);
            sub_2CFDCE0(a1, a2, *(_QWORD *)(a2 - 64), a3, a4);
            return v19;
          }
LABEL_10:
          v75 = v12;
          v17 = strlen(v16);
          v12 = v75;
          v15 = v17;
          goto LABEL_11;
        }
        v69 = v12;
        sub_B19AA0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 40), a3);
        v20 = v69;
        v19 = v21;
        v22 = *(_QWORD *)(v76 + 16);
        if ( v22 && !*(_QWORD *)(v22 + 8) && a2 == *(_QWORD *)(v22 + 24) && (_BYTE)v19 )
        {
          v32 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
          if ( !v32 )
          {
LABEL_73:
            v42 = 0;
            v43 = off_4C5D0D8[0];
            if ( off_4C5D0D8[0] )
            {
              v72 = v20;
              v79 = off_4C5D0D8[0];
              v44 = strlen(off_4C5D0D8[0]);
              v20 = v72;
              v43 = v79;
              v42 = v44;
            }
            sub_B9A090(a2, v43, v42, v20);
            v31 = *(_QWORD *)(a2 - 64);
            goto LABEL_54;
          }
          while ( 1 )
          {
            v33 = *(char **)(v32 + 24);
            v34 = *v33;
            if ( (unsigned __int8)*v33 <= 0x1Cu )
              break;
            if ( v34 == 62 )
            {
              if ( *(_QWORD *)(a2 - 32) != *((_QWORD *)v33 - 4) )
                break;
            }
            else
            {
              if ( v34 != 61 )
                break;
              v35 = *((_QWORD *)v33 + 2);
              if ( !v35 )
                break;
              if ( *(_QWORD *)(v35 + 8) )
                break;
              v36 = *(_BYTE **)(v35 + 24);
              if ( *v36 != 62 )
                break;
              v37 = *((_QWORD *)v36 - 4);
              if ( *(_BYTE *)v37 <= 0x1Cu )
                break;
              v38 = off_4C5D0D8[0];
              if ( off_4C5D0D8[0] )
              {
                v66 = v20;
                v67 = *((_QWORD *)v36 - 4);
                v70 = v32;
                v77 = off_4C5D0D8[0];
                v39 = strlen(off_4C5D0D8[0]);
                v38 = v77;
                v32 = v70;
                v37 = v67;
                v20 = v66;
                v40 = v39;
              }
              else
              {
                v40 = 0;
              }
              if ( !*(_QWORD *)(v37 + 48) && (*(_BYTE *)(v37 + 7) & 0x20) == 0 )
                break;
              v71 = v20;
              v78 = v32;
              v41 = sub_B91F50(v37, v38, v40);
              v32 = v78;
              v20 = v71;
              if ( !v41 )
                break;
            }
            v32 = *(_QWORD *)(v32 + 8);
            if ( !v32 )
              goto LABEL_73;
          }
        }
      }
      else
      {
        if ( v14 != 61 )
          goto LABEL_9;
        v45 = *((_QWORD *)v13 - 4);
        v73 = v12;
        if ( *(_BYTE *)v45 > 0x1Cu )
        {
          v80 = *(_QWORD *)(a2 - 64);
          v46 = sub_B19DB0(*(_QWORD *)a1, v45, (__int64)v13);
          v47 = v80;
          v12 = v73;
          if ( v46 && (v53 = sub_B19DB0(*(_QWORD *)a1, v80, a2), v47 = v80, v12 = v73, v53) )
          {
            sub_B19AA0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 40), a3);
            v12 = v73;
            v47 = v80;
            v48 = v54 ^ 1;
          }
          else
          {
            v48 = 1;
          }
          v49 = *(_QWORD *)(v47 + 16);
          if ( v49 )
          {
            if ( !*(_QWORD *)(v49 + 8) && a2 == *(_QWORD *)(v49 + 24) && !v48 )
            {
              v50 = *(_QWORD *)(v45 + 16);
              if ( !v50 )
              {
LABEL_91:
                v15 = 0;
                v16 = off_4C5D0D8[0];
                if ( !off_4C5D0D8[0] )
                  goto LABEL_11;
                goto LABEL_10;
              }
              while ( 1 )
              {
                v51 = *(_QWORD *)(v50 + 24);
                if ( v47 == v51 )
                  goto LABEL_90;
                v52 = *(_BYTE *)v51;
                if ( *(_BYTE *)v51 <= 0x1Cu )
                  goto LABEL_14;
                if ( v52 != 62 )
                  break;
                v65 = *(_QWORD *)(v51 - 32);
                if ( v45 != v65 || !v65 )
                  goto LABEL_14;
LABEL_90:
                v50 = *(_QWORD *)(v50 + 8);
                if ( !v50 )
                  goto LABEL_91;
              }
              if ( v52 != 63 )
              {
                if ( v52 != 78 )
                  goto LABEL_14;
                v68 = v12;
                v74 = v47;
                v81 = v50;
                sub_98C5F0(v51);
                v12 = v68;
                v47 = v74;
                v50 = v81;
                goto LABEL_90;
              }
              v55 = *(_DWORD *)(v51 + 4) & 0x7FFFFFF;
              v56 = *(_QWORD *)(v51 - 32 * v55);
              if ( !v56
                || v45 != v56
                || *(_BYTE *)(*(_QWORD *)(v45 + 8) + 8LL) != 14
                || *(_BYTE *)(*(_QWORD *)(v47 + 8) + 8LL) != 15
                || (*(_DWORD *)(v51 + 4) & 0x7FFFFFFu) <= 2 )
              {
                goto LABEL_14;
              }
              v57 = *(_QWORD *)(v51 + 32 * (2 - v55));
              v58 = 1 - v55;
              v59 = *(_QWORD *)(a1 + 24);
              v60 = *(_QWORD *)(v51 + 32 * v58);
              if ( v59 )
              {
                if ( v45 != v59 || v60 != *(_QWORD *)(a1 + 32) )
                  goto LABEL_14;
              }
              else
              {
                *(_QWORD *)(a1 + 24) = v45;
                *(_QWORD *)(a1 + 32) = v60;
              }
              if ( *(_BYTE *)v57 != 17 )
                goto LABEL_14;
              v61 = *(_DWORD *)(v57 + 32);
              v62 = *(__int64 **)(v57 + 24);
              if ( v61 > 0x40 )
              {
                v63 = *v62;
              }
              else
              {
                if ( !v61 )
                {
                  v64 = 0;
LABEL_113:
                  if ( *(_BYTE *)(a1 + v64 + 40) )
                    goto LABEL_14;
                  *(_BYTE *)(a1 + v64 + 40) = 1;
                  goto LABEL_90;
                }
                LODWORD(v63) = (__int64)((_QWORD)v62 << (64 - (unsigned __int8)v61)) >> (64 - (unsigned __int8)v61);
              }
              v64 = (int)v63;
              if ( (unsigned int)v63 > 0xF )
                goto LABEL_14;
              goto LABEL_113;
            }
          }
        }
      }
    }
LABEL_14:
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 61 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v23 = *(_QWORD **)(a2 - 8);
    else
      v23 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_2CFDCE0(a1, a2, *v23, a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 63 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v24 = *(_QWORD **)(a2 - 8);
    else
      v24 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_2CFDCE0(a1, a2, *v24, a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 78 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v25 = *(_QWORD **)(a2 - 8);
    else
      v25 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_2CFDCE0(a1, a2, *v25, a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 79 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v26 = *(_QWORD **)(a2 - 8);
    else
      v26 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_2CFDCE0(a1, a2, *v26, a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 93 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v27 = *(_QWORD **)(a2 - 8);
    else
      v27 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_2CFDCE0(a1, a2, *v27, a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 == 94 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v28 = *(_QWORD *)(a2 - 8);
    else
      v28 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    sub_2CFDCE0(a1, a2, *(_QWORD *)(v28 + 32), a3, a4);
    v8 = *(_BYTE *)a2;
  }
  if ( v8 != 85 )
    return 0;
  v30 = *(_QWORD *)(a2 - 32);
  v19 = 0;
  if ( v30
    && !*(_BYTE *)v30
    && *(_QWORD *)(v30 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v30 + 33) & 0x20) != 0
    && *(_DWORD *)(v30 + 36) == 8170 )
  {
    v31 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
LABEL_54:
    sub_2CFDCE0(a1, a2, v31, a3, a4);
  }
  return v19;
}
