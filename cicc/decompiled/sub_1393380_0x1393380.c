// Function: sub_1393380
// Address: 0x1393380
//
void __fastcall sub_1393380(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rbx
  char v4; // dl
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r10
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  int v17; // r12d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r15
  __int64 *v21; // rbx
  __int64 v22; // r12
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rdx
  unsigned __int8 v29; // al
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r13
  unsigned __int64 v56; // r15
  unsigned __int64 v57; // r12
  unsigned __int64 v58; // r14
  __int64 v59; // r13
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rsi
  unsigned int v65; // edi
  __int64 *v66; // rax
  __int64 v67; // r10
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 *v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // rsi
  unsigned int v78; // edi
  __int64 *v79; // rax
  __int64 v80; // r10
  __int64 v81; // rdx
  int v82; // eax
  int v83; // r8d
  __int64 v84; // rax
  int v85; // eax
  int v86; // r8d
  unsigned __int64 v87; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v88; // [rsp+10h] [rbp-A0h]
  __int64 v89; // [rsp+18h] [rbp-98h]
  __int64 v90; // [rsp+20h] [rbp-90h]
  __int64 v91; // [rsp+20h] [rbp-90h]
  __int64 v92; // [rsp+20h] [rbp-90h]
  __int64 v93; // [rsp+28h] [rbp-88h]
  __int64 v94; // [rsp+28h] [rbp-88h]
  __int64 v95; // [rsp+28h] [rbp-88h]
  unsigned __int64 v96; // [rsp+28h] [rbp-88h]
  __int64 v97; // [rsp+30h] [rbp-80h]
  _QWORD v98[2]; // [rsp+38h] [rbp-78h] BYREF
  __int64 v99; // [rsp+48h] [rbp-68h] BYREF
  _QWORD *v100; // [rsp+50h] [rbp-60h] BYREF
  __int64 v101; // [rsp+58h] [rbp-58h]
  _QWORD v102[10]; // [rsp+60h] [rbp-50h] BYREF

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = a1;
  v98[0] = a2;
  v4 = *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (a2 & 4) == 0 )
  {
    if ( v4 >= 0 )
    {
      v13 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v11 = -72;
      goto LABEL_15;
    }
    v14 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
    v16 = v14 + v15;
    if ( *(char *)(v2 + 23) >= 0 )
    {
      if ( (unsigned int)(v16 >> 4) )
        goto LABEL_115;
    }
    else if ( (unsigned int)((v16 - sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
    {
      if ( *(char *)(v2 + 23) < 0 )
      {
        v17 = *(_DWORD *)(sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
        if ( *(char *)(v2 + 23) < 0 )
        {
          v18 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
          v11 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
          v12 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
          v13 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_15;
        }
LABEL_116:
        BUG();
      }
      goto LABEL_115;
    }
    v11 = -72;
    v12 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( v4 >= 0 )
  {
    v13 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = -24;
    goto LABEL_15;
  }
  v5 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v7 = v5 + v6;
  if ( *(char *)(v2 + 23) >= 0 )
  {
    if ( !(unsigned int)(v7 >> 4) )
      goto LABEL_29;
LABEL_115:
    BUG();
  }
  if ( !(unsigned int)((v7 - sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
  {
LABEL_29:
    v11 = -24;
    v12 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_15;
  }
  if ( *(char *)(v2 + 23) >= 0 )
    goto LABEL_115;
  v8 = *(_DWORD *)(sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( *(char *)(v2 + 23) >= 0 )
    goto LABEL_116;
  v9 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v11 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
  v12 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
  v13 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
LABEL_15:
  v20 = (__int64 *)(v13 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
  if ( (__int64 *)(v2 + v11) == v20 )
    goto LABEL_23;
  v21 = (__int64 *)(v2 + v11);
  do
  {
    while ( 1 )
    {
      v22 = *v20;
      if ( *(_BYTE *)(*(_QWORD *)*v20 + 8LL) == 15 )
      {
        v23 = *(_BYTE *)(v22 + 16);
        if ( v23 > 3u )
        {
          if ( v23 == 5 )
          {
            if ( (unsigned int)*(unsigned __int16 *)(v22 + 18) - 51 > 1 )
            {
              v89 &= 0xFFFFFFFF00000000LL;
              if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), v22, v89, 0) )
                sub_1391610(a1, v22, v28);
            }
          }
          else
          {
            v93 &= 0xFFFFFFFF00000000LL;
            sub_13848E0(*(_QWORD *)(a1 + 24), v22, v93, 0);
          }
          goto LABEL_17;
        }
        v90 = *(_QWORD *)(a1 + 24);
        v24 = sub_14C81A0(*v20);
        v97 &= 0xFFFFFFFF00000000LL;
        if ( (unsigned __int8)sub_13848E0(v90, v22, v97, v24) )
          break;
      }
LABEL_17:
      v20 += 3;
      if ( v21 == v20 )
        goto LABEL_22;
    }
    v20 += 3;
    v91 = *(_QWORD *)(a1 + 24);
    v26 = sub_14C8160(v91, v22, v25);
    v88 = v88 & 0xFFFFFFFF00000000LL | 1;
    sub_13848E0(v91, v22, 1u, v26);
  }
  while ( v21 != v20 );
LABEL_22:
  v3 = a1;
LABEL_23:
  if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 )
  {
    v29 = *(_BYTE *)(v2 + 16);
    if ( v29 > 3u )
    {
      if ( v29 == 5 )
      {
        if ( (unsigned int)*(unsigned __int16 *)(v2 + 18) - 51 > 1
          && (unsigned __int8)sub_13848E0(*(_QWORD *)(v3 + 24), v2, 0, 0) )
        {
          sub_1391610(v3, v2, v48);
        }
        goto LABEL_24;
      }
    }
    else if ( v2 )
    {
      v30 = *(_QWORD *)(v3 + 24);
      v31 = sub_14C81A0(v2);
      v32 = v30;
      if ( (unsigned __int8)sub_13848E0(v30, v2, 0, v31) )
      {
        v34 = *(_QWORD *)(v3 + 24);
        v35 = sub_14C8160(v32, v2, v33);
        sub_13848E0(v34, v2, 1u, v35);
      }
      goto LABEL_24;
    }
    sub_13848E0(*(_QWORD *)(v3 + 24), v2, 0, 0);
  }
LABEL_24:
  if ( !(unsigned __int8)sub_140B160(v2, *(_QWORD *)(v3 + 16), 0) )
  {
    v27 = sub_140B650(v2, *(_QWORD *)(v3 + 16));
    if ( !v27 )
    {
      v100 = v102;
      v101 = 0x400000000LL;
      v36 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v98[0] & 4) != 0 )
      {
        v37 = *(_QWORD *)(v36 - 24);
        if ( *(_BYTE *)(v37 + 16) )
        {
LABEL_43:
          v38 = 0xFFFFFFFFLL;
          v39 = (__int64 *)(v36 + 56);
          if ( (unsigned __int8)sub_1560260(v36 + 56, 0xFFFFFFFFLL, 36) )
            goto LABEL_44;
          if ( *(char *)(v36 + 23) >= 0 )
            goto LABEL_119;
          v49 = sub_1648A40(v36);
          v95 = v50 + v49;
          v51 = 0;
          if ( *(char *)(v36 + 23) < 0 )
            v51 = sub_1648A40(v36);
          if ( !(unsigned int)((v95 - v51) >> 4) )
          {
LABEL_119:
            v52 = *(_QWORD *)(v36 - 24);
            if ( !*(_BYTE *)(v52 + 16) )
            {
              v39 = &v99;
              v38 = 0xFFFFFFFFLL;
              v99 = *(_QWORD *)(v52 + 112);
              if ( (unsigned __int8)sub_1560260(&v99, 0xFFFFFFFFLL, 36) )
                goto LABEL_44;
            }
          }
          v38 = 0xFFFFFFFFLL;
          v39 = (__int64 *)(v36 + 56);
          if ( (unsigned __int8)sub_1560260(v36 + 56, 0xFFFFFFFFLL, 37) )
            goto LABEL_44;
          if ( *(char *)(v36 + 23) < 0 )
          {
            v53 = sub_1648A40(v36);
            v55 = v53 + v54;
            if ( *(char *)(v36 + 23) < 0 )
              v27 = sub_1648A40(v36);
            if ( v27 != v55 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v27 + 8LL) <= 1u )
              {
                v27 += 16;
                if ( v55 == v27 )
                  goto LABEL_104;
              }
              goto LABEL_84;
            }
          }
LABEL_104:
          v47 = *(_QWORD *)(v36 - 24);
          if ( *(_BYTE *)(v47 + 16) )
          {
LABEL_84:
            v39 = v98;
            v56 = sub_1389B50(v98);
            v57 = (v98[0] & 0xFFFFFFFFFFFFFFF8LL)
                - 24LL * (*(_DWORD *)((v98[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
            if ( v56 != v57 )
            {
              v96 = v2;
              v58 = v56;
              do
              {
                v59 = *(_QWORD *)v57;
                if ( *(_BYTE *)(**(_QWORD **)v57 + 8LL) == 15 )
                {
                  v60 = *(_QWORD *)(v3 + 24);
                  v61 = sub_14C8190();
                  v62 = *(unsigned int *)(v60 + 24);
                  v63 = v61;
                  if ( !(_DWORD)v62 )
                    goto LABEL_114;
                  v64 = *(_QWORD *)(v60 + 8);
                  v65 = (v62 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
                  v66 = (__int64 *)(v64 + 32LL * v65);
                  v67 = *v66;
                  if ( *v66 != v59 )
                  {
                    v82 = 1;
                    while ( v67 != -8 )
                    {
                      v83 = v82 + 1;
                      v84 = ((_DWORD)v62 - 1) & (v65 + v82);
                      v65 = v84;
                      v66 = (__int64 *)(v64 + 32 * v84);
                      v67 = *v66;
                      if ( v59 == *v66 )
                        goto LABEL_90;
                      v82 = v83;
                    }
LABEL_114:
                    BUG();
                  }
LABEL_90:
                  if ( v66 == (__int64 *)(v64 + 32 * v62) )
                    goto LABEL_114;
                  v68 = v66[1];
                  if ( !(-1227133513 * (unsigned int)((v66[2] - v68) >> 3)) )
                    goto LABEL_114;
                  *(_QWORD *)(v68 + 48) |= v63;
                  v92 = *(_QWORD *)(v3 + 24);
                  v69 = sub_14C8160(v92, 0x6DB6DB6DB6DB6DB7LL, v68);
                  v39 = (__int64 *)v92;
                  v38 = v59;
                  v87 = v87 & 0xFFFFFFFF00000000LL | 1;
                  sub_13848E0(v92, v59, 1u, v69);
                }
                v57 += 24LL;
              }
              while ( v58 != v57 );
              v2 = v96;
            }
            goto LABEL_44;
          }
          goto LABEL_65;
        }
      }
      else
      {
        v37 = *(_QWORD *)(v36 - 72);
        if ( *(_BYTE *)(v37 + 16) )
          goto LABEL_50;
      }
      v102[0] = v37;
      LODWORD(v101) = 1;
      if ( (unsigned __int8)sub_13960C0(v3, v98[0], &v100) )
      {
LABEL_45:
        if ( v100 != v102 )
          _libc_free((unsigned __int64)v100);
        return;
      }
      v36 = v98[0] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v98[0] & 4) != 0 )
        goto LABEL_43;
LABEL_50:
      v38 = 0xFFFFFFFFLL;
      v39 = (__int64 *)(v36 + 56);
      if ( (unsigned __int8)sub_1560260(v36 + 56, 0xFFFFFFFFLL, 36) )
        goto LABEL_44;
      if ( *(char *)(v36 + 23) >= 0 )
        goto LABEL_120;
      v40 = sub_1648A40(v36);
      v94 = v41 + v40;
      v42 = 0;
      if ( *(char *)(v36 + 23) < 0 )
        v42 = sub_1648A40(v36);
      if ( !(unsigned int)((v94 - v42) >> 4) )
      {
LABEL_120:
        v43 = *(_QWORD *)(v36 - 72);
        if ( !*(_BYTE *)(v43 + 16) )
        {
          v39 = &v99;
          v38 = 0xFFFFFFFFLL;
          v99 = *(_QWORD *)(v43 + 112);
          if ( (unsigned __int8)sub_1560260(&v99, 0xFFFFFFFFLL, 36) )
            goto LABEL_44;
        }
      }
      v38 = 0xFFFFFFFFLL;
      v39 = (__int64 *)(v36 + 56);
      if ( (unsigned __int8)sub_1560260(v36 + 56, 0xFFFFFFFFLL, 37) )
        goto LABEL_44;
      if ( *(char *)(v36 + 23) < 0 )
      {
        v44 = sub_1648A40(v36);
        v46 = v44 + v45;
        if ( *(char *)(v36 + 23) < 0 )
          v27 = sub_1648A40(v36);
        if ( v27 != v46 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v27 + 8LL) <= 1u )
          {
            v27 += 16;
            if ( v46 == v27 )
              goto LABEL_64;
          }
          goto LABEL_84;
        }
      }
LABEL_64:
      v47 = *(_QWORD *)(v36 - 72);
      if ( *(_BYTE *)(v47 + 16) )
        goto LABEL_84;
LABEL_65:
      v39 = &v99;
      v38 = 0xFFFFFFFFLL;
      v99 = *(_QWORD *)(v47 + 112);
      if ( !(unsigned __int8)sub_1560260(&v99, 0xFFFFFFFFLL, 37) )
        goto LABEL_84;
LABEL_44:
      if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 )
      {
        v70 = (__int64 *)((v98[0] & 0xFFFFFFFFFFFFFFF8LL) - 72);
        v71 = v98[0] & 4;
        if ( (v98[0] & 4) != 0 )
          v70 = (__int64 *)((v98[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
        v72 = *v70;
        if ( *(_BYTE *)(v72 + 16)
          || (v38 = 0, v39 = (__int64 *)(v72 + 112), !(unsigned __int8)sub_1560260(v72 + 112, 0, 20)) )
        {
          v73 = *(_QWORD *)(v3 + 24);
          v74 = sub_14C8160(v39, v38, v71);
          v75 = *(unsigned int *)(v73 + 24);
          v76 = v74;
          if ( !(_DWORD)v75 )
            goto LABEL_114;
          v77 = *(_QWORD *)(v73 + 8);
          v78 = (v75 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
          v79 = (__int64 *)(v77 + 32LL * v78);
          v80 = *v79;
          if ( v2 != *v79 )
          {
            v85 = 1;
            while ( v80 != -8 )
            {
              v86 = v85 + 1;
              v78 = (v75 - 1) & (v85 + v78);
              v79 = (__int64 *)(v77 + 32LL * v78);
              v80 = *v79;
              if ( v2 == *v79 )
                goto LABEL_99;
              v85 = v86;
            }
            goto LABEL_114;
          }
LABEL_99:
          if ( v79 == (__int64 *)(v77 + 32 * v75) )
            goto LABEL_114;
          v81 = v79[1];
          if ( !(-1227133513 * (unsigned int)((v79[2] - v81) >> 3)) )
            goto LABEL_114;
          *(_QWORD *)(v81 + 48) |= v76;
        }
      }
      goto LABEL_45;
    }
  }
}
