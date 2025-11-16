// Function: sub_1CB4290
// Address: 0x1cb4290
//
__int64 __fastcall sub_1CB4290(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  char v8; // al
  __int64 v9; // r8
  size_t v10; // rdx
  char *v11; // r15
  size_t v12; // rax
  __int64 v13; // r8
  unsigned __int8 v14; // al
  size_t v15; // rdx
  const char *v16; // r15
  const char *v17; // rsi
  unsigned int v18; // r15d
  unsigned int v19; // eax
  __int64 v20; // rdi
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  unsigned __int8 v31; // dl
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // r8
  size_t v35; // rdx
  char *v36; // rsi
  size_t v37; // rax
  size_t v38; // rdx
  char *v39; // rsi
  __int64 v40; // r15
  char v41; // al
  __int64 v42; // r8
  char v43; // dl
  __int64 v44; // rdi
  _QWORD *v45; // rax
  _QWORD *v46; // r8
  __int64 i; // rdx
  _QWORD *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdi
  unsigned __int8 v51; // al
  char v52; // al
  bool v53; // al
  __int64 v54; // rsi
  __int64 v55; // rcx
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rax
  unsigned int v60; // esi
  __int64 *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // [rsp+0h] [rbp-50h]
  char *v65; // [rsp+8h] [rbp-48h]
  __int64 v66; // [rsp+8h] [rbp-48h]
  _QWORD *v67; // [rsp+8h] [rbp-48h]
  __int64 v68; // [rsp+10h] [rbp-40h]
  __int64 v69; // [rsp+10h] [rbp-40h]
  __int64 v70; // [rsp+10h] [rbp-40h]
  char v71; // [rsp+10h] [rbp-40h]
  __int64 v72; // [rsp+10h] [rbp-40h]
  __int64 v73; // [rsp+18h] [rbp-38h]
  __int64 v74; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_OWORD *)(a1 + 40) = 0;
  v8 = *(_BYTE *)(a2 + 16);
  if ( v8 == 55 )
  {
    v9 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v9 + 16) <= 0x17u )
      return 0;
    v10 = 0;
    v11 = off_4CD4978[0];
    if ( off_4CD4978[0] )
    {
      v73 = *(_QWORD *)(a2 - 24);
      v12 = strlen(off_4CD4978[0]);
      v9 = v73;
      v10 = v12;
    }
    if ( !*(_QWORD *)(v9 + 48) && *(__int16 *)(v9 + 18) >= 0 )
      return 0;
    v74 = sub_1625940(v9, v11, v10);
    if ( v74 )
    {
      v13 = *(_QWORD *)(a2 - 48);
      v14 = *(_BYTE *)(v13 + 16);
      if ( v14 <= 0x17u )
      {
        v68 = *(_QWORD *)(a2 - 48);
        if ( v14 != 17 )
        {
LABEL_9:
          v15 = 0;
          v16 = off_4CD4978[0];
          if ( !off_4CD4978[0] )
          {
LABEL_11:
            v17 = v16;
            v18 = 1;
            sub_1626100(a2, v17, v15, v74);
            sub_1CB4980(a1, a2, *(_QWORD *)(a2 - 48), a3, a4);
            return v18;
          }
LABEL_10:
          v15 = strlen(v16);
          goto LABEL_11;
        }
        LOBYTE(v19) = sub_15CCCD0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 40), a3);
        v18 = v19;
        v20 = *(_QWORD *)(v68 + 8);
        if ( v20 && !*(_QWORD *)(v20 + 8) && (_QWORD *)a2 == sub_1648700(v20) && (_BYTE)v18 )
        {
          v69 = *(_QWORD *)(*(_QWORD *)(a2 - 24) + 8LL);
          if ( !v69 )
          {
LABEL_71:
            v38 = 0;
            v39 = off_4CD4978[0];
            if ( off_4CD4978[0] )
            {
              v39 = off_4CD4978[0];
              v38 = strlen(off_4CD4978[0]);
            }
            sub_1626100(a2, v39, v38, v74);
            v29 = *(_QWORD *)(a2 - 48);
            goto LABEL_54;
          }
          while ( 1 )
          {
            v30 = sub_1648700(v69);
            v31 = *((_BYTE *)v30 + 16);
            if ( v31 <= 0x17u )
              break;
            if ( v31 == 55 )
            {
              if ( *(_QWORD *)(a2 - 24) != *(v30 - 3) )
                break;
            }
            else
            {
              if ( v31 != 54 )
                break;
              v32 = v30[1];
              if ( !v32 )
                break;
              if ( *(_QWORD *)(v32 + 8) )
                break;
              v33 = sub_1648700(v32);
              if ( *((_BYTE *)v33 + 16) != 55 )
                break;
              v34 = *(v33 - 3);
              if ( *(_BYTE *)(v34 + 16) <= 0x17u )
                break;
              v35 = 0;
              v36 = off_4CD4978[0];
              if ( off_4CD4978[0] )
              {
                v64 = *(v33 - 3);
                v65 = off_4CD4978[0];
                v37 = strlen(off_4CD4978[0]);
                v34 = v64;
                v36 = v65;
                v35 = v37;
              }
              if ( !*(_QWORD *)(v34 + 48) && *(__int16 *)(v34 + 18) >= 0 || !sub_1625940(v34, v36, v35) )
                break;
            }
            v69 = *(_QWORD *)(v69 + 8);
            if ( !v69 )
              goto LABEL_71;
          }
        }
      }
      else
      {
        if ( v14 != 54 )
          goto LABEL_9;
        v40 = *(_QWORD *)(v13 - 24);
        if ( *(_BYTE *)(v40 + 16) > 0x17u )
        {
          v70 = *(_QWORD *)(a2 - 48);
          v41 = sub_15CCEE0(*(_QWORD *)a1, *(_QWORD *)(v13 - 24), v70);
          v42 = v70;
          if ( v41 && (v52 = sub_15CCEE0(*(_QWORD *)a1, v70, a2), v42 = v70, v52) )
          {
            v53 = sub_15CCCD0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 40), a3);
            v42 = v70;
            v43 = !v53;
          }
          else
          {
            v43 = 1;
          }
          v44 = *(_QWORD *)(v42 + 8);
          if ( v44 )
          {
            v66 = v42;
            v71 = v43;
            if ( !*(_QWORD *)(v44 + 8) )
            {
              v45 = sub_1648700(v44);
              v46 = (_QWORD *)v66;
              if ( (_QWORD *)a2 == v45 && !v71 )
              {
                for ( i = *(_QWORD *)(v40 + 8); i; i = *(_QWORD *)(v49 + 8) )
                {
                  v67 = v46;
                  v72 = i;
                  v48 = sub_1648700(i);
                  v46 = v67;
                  v49 = v72;
                  v50 = (__int64)v48;
                  if ( v67 != v48 )
                  {
                    v51 = *((_BYTE *)v48 + 16);
                    if ( v51 <= 0x17u )
                      goto LABEL_16;
                    switch ( v51 )
                    {
                      case '7':
                        v63 = *(_QWORD *)(v50 - 24);
                        if ( !v63 || v40 != v63 )
                          goto LABEL_16;
                        break;
                      case '8':
                        v54 = *(_DWORD *)(v50 + 20) & 0xFFFFFFF;
                        v55 = *(_QWORD *)(v50 - 24 * v54);
                        if ( v40 != v55
                          || !v55
                          || *(_BYTE *)(*(_QWORD *)v40 + 8LL) != 15
                          || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v40 + 24LL) + 8LL) != 13
                          || (*(_DWORD *)(v50 + 20) & 0xFFFFFFFu) <= 2 )
                        {
                          goto LABEL_16;
                        }
                        v56 = *(_QWORD *)(v50 + 24 * (2 - v54));
                        v57 = 1 - v54;
                        v58 = *(_QWORD *)(a1 + 24);
                        v59 = *(_QWORD *)(v50 + 24 * v57);
                        if ( v58 )
                        {
                          if ( v40 != v58 || v59 != *(_QWORD *)(a1 + 32) )
                            goto LABEL_16;
                        }
                        else
                        {
                          *(_QWORD *)(a1 + 24) = v40;
                          *(_QWORD *)(a1 + 32) = v59;
                        }
                        if ( *(_BYTE *)(v56 + 16) != 13 )
                          goto LABEL_16;
                        v60 = *(_DWORD *)(v56 + 32);
                        v61 = *(__int64 **)(v56 + 24);
                        v62 = v60 > 0x40
                            ? *v61
                            : (__int64)((_QWORD)v61 << (64 - (unsigned __int8)v60)) >> (64 - (unsigned __int8)v60);
                        if ( (unsigned int)v62 > 0xF || *(_BYTE *)(a1 + (int)v62 + 40) )
                          goto LABEL_16;
                        *(_BYTE *)(a1 + (int)v62 + 40) = 1;
                        break;
                      case 'G':
                        sub_14ADF20(v50);
                        v46 = v67;
                        v49 = v72;
                        break;
                      default:
                        goto LABEL_16;
                    }
                  }
                }
                v15 = 0;
                v16 = off_4CD4978[0];
                if ( !off_4CD4978[0] )
                  goto LABEL_11;
                goto LABEL_10;
              }
            }
          }
        }
      }
    }
LABEL_16:
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 54 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v21 = *(_QWORD **)(a2 - 8);
    else
      v21 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_1CB4980(a1, a2, *v21, a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 56 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v22 = *(_QWORD **)(a2 - 8);
    else
      v22 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_1CB4980(a1, a2, *v22, a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 71 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v23 = *(_QWORD **)(a2 - 8);
    else
      v23 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_1CB4980(a1, a2, *v23, a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 72 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v24 = *(_QWORD **)(a2 - 8);
    else
      v24 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_1CB4980(a1, a2, *v24, a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 86 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v25 = *(_QWORD **)(a2 - 8);
    else
      v25 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    sub_1CB4980(a1, a2, *v25, a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 == 87 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v26 = *(_QWORD *)(a2 - 8);
    else
      v26 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    sub_1CB4980(a1, a2, *(_QWORD *)(v26 + 24), a3, a4);
    v8 = *(_BYTE *)(a2 + 16);
  }
  if ( v8 != 78 )
    return 0;
  v28 = *(_QWORD *)(a2 - 24);
  v18 = 0;
  if ( !*(_BYTE *)(v28 + 16) && (*(_BYTE *)(v28 + 33) & 0x20) != 0 && *(_DWORD *)(v28 + 36) == 3660 )
  {
    v29 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
LABEL_54:
    sub_1CB4980(a1, a2, v29, a3, a4);
  }
  return v18;
}
