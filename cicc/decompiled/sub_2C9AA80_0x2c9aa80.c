// Function: sub_2C9AA80
// Address: 0x2c9aa80
//
char *__fastcall sub_2C9AA80(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v6; // rdx
  unsigned __int8 *v7; // r15
  __int64 v8; // r10
  __int64 v9; // rcx
  int v10; // r8d
  unsigned int i; // eax
  unsigned __int8 **v12; // rsi
  unsigned int v13; // eax
  int v14; // edx
  __int64 v15; // r10
  unsigned __int8 v16; // al
  char *v17; // r14
  unsigned __int8 *v18; // r13
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r10
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // r10
  __int64 v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // r10
  __int64 v34; // rax
  const char *v35; // r10
  __int64 v36; // rdi
  __int64 v37; // rax
  const char *v38; // rax
  bool v39; // al
  const char *v40; // r10
  bool v41; // al
  __int64 v42; // r10
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r15
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rax
  __int64 v51; // r10
  __int64 v52; // r8
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r15
  _BYTE *v60; // rdx
  __int64 v61; // rbx
  __int64 *v62; // rax
  _BYTE *v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned int *v66; // rcx
  unsigned int *v67; // r14
  __int64 v68; // [rsp+8h] [rbp-168h]
  __int64 v69; // [rsp+8h] [rbp-168h]
  __int64 v70; // [rsp+10h] [rbp-160h]
  __int64 v71; // [rsp+10h] [rbp-160h]
  __int64 v72; // [rsp+10h] [rbp-160h]
  __int64 v73; // [rsp+10h] [rbp-160h]
  __int64 v74; // [rsp+10h] [rbp-160h]
  const char *v75; // [rsp+10h] [rbp-160h]
  __int64 v76; // [rsp+10h] [rbp-160h]
  __int64 v77; // [rsp+10h] [rbp-160h]
  __int64 v78; // [rsp+10h] [rbp-160h]
  __int64 v79; // [rsp+10h] [rbp-160h]
  __int64 v80; // [rsp+18h] [rbp-158h]
  __int64 v81; // [rsp+20h] [rbp-150h]
  __int64 v82; // [rsp+20h] [rbp-150h]
  __int64 v83; // [rsp+20h] [rbp-150h]
  const char *v84; // [rsp+20h] [rbp-150h]
  __int64 *v85; // [rsp+20h] [rbp-150h]
  __int64 v86; // [rsp+28h] [rbp-148h]
  char *v87; // [rsp+28h] [rbp-148h]
  __int64 *v88; // [rsp+28h] [rbp-148h]
  unsigned int *v89; // [rsp+28h] [rbp-148h]
  __int64 v90; // [rsp+38h] [rbp-138h] BYREF
  _QWORD v91[2]; // [rsp+40h] [rbp-130h] BYREF
  const char *v92; // [rsp+50h] [rbp-120h] BYREF
  __int64 v93; // [rsp+58h] [rbp-118h]
  char v94; // [rsp+70h] [rbp-100h]
  char v95; // [rsp+71h] [rbp-FFh]
  const char *v96; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v97; // [rsp+88h] [rbp-E8h]
  __int16 v98; // [rsp+A0h] [rbp-D0h]
  unsigned int *v99; // [rsp+B0h] [rbp-C0h] BYREF
  int v100; // [rsp+B8h] [rbp-B8h]
  char v101; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v102; // [rsp+E8h] [rbp-88h]
  __int64 v103; // [rsp+F0h] [rbp-80h]
  __int64 v104; // [rsp+100h] [rbp-70h]
  __int64 v105; // [rsp+108h] [rbp-68h]
  void *v106; // [rsp+130h] [rbp-40h]

  v6 = *(unsigned int *)(a1 + 136);
  v7 = (unsigned __int8 *)*a2;
  v8 = a2[1];
  v9 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v6 )
  {
    v10 = 1;
    for ( i = (v6 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = (v6 - 1) & v13 )
    {
      v12 = (unsigned __int8 **)(v9 + 32LL * i);
      if ( v7 == *v12 && (unsigned __int8 *)v8 == v12[1] )
        break;
      if ( *v12 == (unsigned __int8 *)-4096LL && v12[1] == (unsigned __int8 *)-4096LL )
        goto LABEL_7;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != (unsigned __int8 **)(v9 + 32 * v6) )
    {
      v17 = (char *)v12[2];
      v18 = v12[3];
      if ( v17
        && ((unsigned __int8)*v17 <= 0x1Cu
         || *v7 <= 0x1Cu
         || (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 200), (__int64)v12[2], (__int64)v7)) )
      {
        *a3 = (__int64)v18;
        return v17;
      }
      return 0;
    }
  }
LABEL_7:
  v86 = v8;
  if ( *v7 <= 0x1Cu )
    return 0;
  v80 = a1 + 112;
  sub_23D0AB0((__int64)&v99, (__int64)v7, 0, 0, 0);
  v14 = *v7;
  v15 = v86;
  v16 = *v7;
  if ( (unsigned int)(v14 - 67) <= 0xC )
  {
    v17 = 0;
    if ( (_BYTE)v14 == 67 )
      goto LABEL_10;
    v20 = *((_QWORD *)v7 - 4);
    if ( *(_BYTE *)v20 <= 0x1Cu )
      goto LABEL_31;
    if ( *(_QWORD *)(v20 + 40) == *((_QWORD *)v7 + 5) )
      goto LABEL_32;
    v17 = *(char **)(v20 + 16);
    if ( !v17 )
    {
LABEL_31:
      v21 = *a3;
      v22 = sub_2C9A8E0(v80, a2);
      *v22 = (__int64)v17;
      v22[1] = v21;
      goto LABEL_10;
    }
    if ( !*((_QWORD *)v17 + 1) )
    {
LABEL_32:
      v92 = (const char *)*((_QWORD *)v7 - 4);
      v93 = a2[1];
      v23 = sub_2C9AA80(a1, &v92, a3);
      v17 = (char *)v23;
      if ( v23 )
      {
        v96 = "newCast";
        v98 = 259;
        v17 = (char *)sub_2C91010(
                        (__int64 *)&v99,
                        (unsigned int)*v7 - 29,
                        v23,
                        *((_QWORD *)v7 + 1),
                        (__int64)&v96,
                        0,
                        v91[0],
                        0);
      }
      goto LABEL_31;
    }
LABEL_30:
    v17 = 0;
    goto LABEL_31;
  }
  v17 = 0;
  if ( (unsigned int)(v14 - 42) > 0x11 )
    goto LABEL_10;
  v17 = (char *)*((_QWORD *)v7 - 8);
  v87 = (char *)*((_QWORD *)v7 - 4);
  if ( v16 != 42 )
  {
    if ( (int)qword_5011F68 <= 2 || v16 != 46 )
    {
      v17 = 0;
      goto LABEL_10;
    }
    if ( *v87 == 17 )
    {
      v78 = v15;
      v85 = sub_DD8400(*(_QWORD *)(a1 + 184), (__int64)v87);
      if ( !sub_D968A0((__int64)v85) )
      {
        sub_2C90D50(*(_QWORD *)(v85[4] + 24), *(_DWORD *)(v85[4] + 32));
        v50 = sub_2C90D50(*(_QWORD *)(*(_QWORD *)(v78 + 32) + 24LL), *(_DWORD *)(*(_QWORD *)(v78 + 32) + 32LL));
        v53 = v50 % v52;
        v54 = v50 / v52;
        if ( !v53
          && (unsigned __int8)*v17 > 0x1Cu
          && (*((_QWORD *)v17 + 5) == *((_QWORD *)v7 + 5) || (v55 = *((_QWORD *)v17 + 2)) != 0 && !*(_QWORD *)(v55 + 8)) )
        {
          v79 = v54;
          v56 = *(_QWORD *)(a1 + 184);
          v57 = sub_D95540(v51);
          v91[0] = v17;
          v90 = 0;
          v91[1] = sub_DA2C50(v56, v57, v79, 0);
          v58 = sub_2C9AA80(a1, v91, &v90);
          v17 = (char *)v58;
          if ( v58 )
          {
            v95 = 1;
            v94 = 3;
            v92 = "newMul";
            v59 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, char *, _QWORD, _QWORD))(*(_QWORD *)v104 + 32LL))(
                    v104,
                    17,
                    v58,
                    v87,
                    0,
                    0);
            if ( !v59 )
            {
              v98 = 257;
              v59 = sub_B504D0(17, (__int64)v17, (__int64)v87, (__int64)&v96, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v105 + 16LL))(
                v105,
                v59,
                &v92,
                v102,
                v103);
              v66 = v99;
              v67 = &v99[4 * v100];
              while ( v66 != v67 )
              {
                v89 = v66;
                sub_B99FD0(v59, *v66, *((_QWORD *)v66 + 1));
                v66 = v89 + 4;
              }
            }
            v17 = (char *)v59;
          }
          if ( v90 )
            *a3 = (__int64)sub_DCA690(*(__int64 **)(a1 + 184), (__int64)v85, v90, 0, 0);
          goto LABEL_31;
        }
      }
    }
    goto LABEL_30;
  }
  v81 = v15;
  v24 = sub_DD8400(*(_QWORD *)(a1 + 184), (__int64)v17);
  v25 = v81;
  v26 = (__int64)v24;
  if ( *((_WORD *)v24 + 12) )
  {
    sub_2C95190(&v96, (__int64)v24, *(__int64 **)(a1 + 184));
    v26 = (__int64)v96;
    v25 = v81;
  }
  v70 = v25;
  v82 = sub_D95540(v26);
  v27 = sub_D95540(v70);
  v28 = v70;
  if ( v82 != v27 )
  {
    v29 = v70;
    v83 = v70;
    v71 = *(_QWORD *)(a1 + 184);
    v30 = sub_D95540(v29);
    v31 = sub_DC5000(v71, v26, v30, 0);
    v28 = v83;
    v26 = (__int64)v31;
  }
  if ( v28 == v26 )
  {
    if ( *v17 != 17 )
    {
      v92 = v17;
      v93 = v28;
      v63 = (_BYTE *)sub_2C9AA80(a1, &v92, a3);
      if ( v63 )
      {
        v96 = "newAdd";
        v98 = 259;
        v87 = (char *)sub_929C50(&v99, v63, v87, (__int64)&v96, 0, 0);
      }
      else
      {
        v87 = 0;
      }
    }
    v61 = *a3;
    v62 = sub_2C9A8E0(v80, a2);
    v62[1] = v61;
    *v62 = (__int64)v87;
    v17 = v87;
    goto LABEL_10;
  }
  v72 = v28;
  v32 = sub_DD8400(*(_QWORD *)(a1 + 184), (__int64)v87);
  v33 = v72;
  v84 = (const char *)v32;
  if ( *((_WORD *)v32 + 12) )
  {
    sub_2C95190(&v96, (__int64)v32, *(__int64 **)(a1 + 184));
    v33 = v72;
    v84 = v96;
  }
  v68 = v33;
  v73 = sub_D95540((__int64)v84);
  v34 = sub_D95540(v68);
  v35 = (const char *)v68;
  if ( v73 != v34 )
  {
    v36 = v68;
    v74 = v68;
    v69 = *(_QWORD *)(a1 + 184);
    v37 = sub_D95540(v36);
    v38 = (const char *)sub_DC5000(v69, (__int64)v84, v37, 0);
    v35 = (const char *)v74;
    v84 = v38;
  }
  if ( v35 == v84 )
  {
    if ( *v87 == 17 )
      goto LABEL_31;
    v92 = v87;
    v93 = (__int64)v35;
    v60 = (_BYTE *)sub_2C9AA80(a1, &v92, a3);
    if ( v60 )
    {
      v96 = "newAdd";
      v98 = 259;
      v17 = (char *)sub_929C50(&v99, v17, v60, (__int64)&v96, 0, 0);
      goto LABEL_31;
    }
    goto LABEL_30;
  }
  v75 = v35;
  v91[0] = 0;
  v92 = 0;
  v39 = sub_D968A0(v26);
  v40 = v75;
  if ( v39 || (v96 = v17, v97 = v26, v64 = sub_2C9AA80(a1, &v96, v91), v40 = v75, !v64) )
    v91[0] = v26;
  else
    v17 = (char *)v64;
  v76 = (__int64)v40;
  v41 = sub_D968A0((__int64)v84);
  v42 = v76;
  if ( v41 )
  {
    v92 = v84;
    if ( v26 != v91[0] )
    {
LABEL_48:
      v77 = v42;
      v96 = "newAdd";
      v98 = 259;
      v17 = (char *)sub_929C50(&v99, v17, v87, (__int64)&v96, 0, 0);
      v88 = *(__int64 **)(a1 + 184);
      v43 = sub_DC7ED0(v88, v26, (__int64)v84, 0, 0);
      v44 = sub_DCC810(v88, v77, (__int64)v43, 0, 0);
      v45 = v91[0];
      *a3 = (__int64)v44;
      v46 = (__int64)v44;
      if ( v45 )
      {
        v47 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v44, v45, 0, 0);
        *a3 = (__int64)v47;
        v46 = (__int64)v47;
      }
      if ( v92 )
      {
        v48 = sub_DC7ED0(*(__int64 **)(a1 + 184), v46, (__int64)v92, 0, 0);
        *a3 = (__int64)v48;
        v46 = (__int64)v48;
      }
      goto LABEL_52;
    }
  }
  else
  {
    v96 = v87;
    v97 = (__int64)v84;
    v65 = sub_2C9AA80(a1, &v96, &v92);
    v42 = v76;
    if ( v65 )
      v87 = (char *)v65;
    else
      v92 = v84;
    if ( v26 != v91[0] || v84 != v92 )
      goto LABEL_48;
  }
  v46 = *a3;
  v17 = 0;
LABEL_52:
  v49 = sub_2C9A8E0(v80, a2);
  *v49 = (__int64)v17;
  v49[1] = v46;
LABEL_10:
  nullsub_61();
  v106 = &unk_49DA100;
  nullsub_63();
  if ( v99 != (unsigned int *)&v101 )
    _libc_free((unsigned __int64)v99);
  return v17;
}
