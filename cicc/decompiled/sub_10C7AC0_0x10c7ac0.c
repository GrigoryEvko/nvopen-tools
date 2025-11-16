// Function: sub_10C7AC0
// Address: 0x10c7ac0
//
__int64 __fastcall sub_10C7AC0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  int v3; // eax
  unsigned __int8 *v4; // rbx
  __int64 v5; // r14
  _QWORD *v6; // r12
  __int64 v8; // r15
  __int64 v9; // r11
  int v10; // ecx
  _BYTE *v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  int v16; // r15d
  __int64 v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // r14
  unsigned int v20; // r12d
  unsigned int v21; // eax
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r11
  __int64 v29; // r10
  _QWORD *v30; // rax
  _QWORD *v31; // r15
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r11
  _QWORD *v37; // rax
  char v38; // al
  __int64 v39; // r10
  __int64 v40; // r11
  __int64 v41; // rdx
  int v42; // r15d
  __int64 v43; // rcx
  __int64 v45; // rbx
  __int64 v46; // r13
  __int64 v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  char v50; // al
  __int64 v51; // r11
  __int64 v52; // rdx
  int v53; // r15d
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  bool v60; // al
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 *v65; // r14
  const char *v66; // rax
  __int64 *v67; // rdx
  __int64 v68; // rax
  __int64 *v69; // [rsp+8h] [rbp-F8h]
  __int64 v70; // [rsp+10h] [rbp-F0h]
  __int64 v71; // [rsp+10h] [rbp-F0h]
  __int64 v72; // [rsp+10h] [rbp-F0h]
  __int64 v73; // [rsp+18h] [rbp-E8h]
  __int64 *v74; // [rsp+18h] [rbp-E8h]
  __int64 v75; // [rsp+18h] [rbp-E8h]
  __int64 v76; // [rsp+18h] [rbp-E8h]
  _BYTE *v77; // [rsp+18h] [rbp-E8h]
  _BYTE *v78; // [rsp+20h] [rbp-E0h]
  __int64 v79; // [rsp+20h] [rbp-E0h]
  __int64 v80; // [rsp+20h] [rbp-E0h]
  __int64 v81; // [rsp+20h] [rbp-E0h]
  __int64 v82; // [rsp+20h] [rbp-E0h]
  __int64 v83; // [rsp+20h] [rbp-E0h]
  __int64 v84; // [rsp+20h] [rbp-E0h]
  __int64 v85; // [rsp+20h] [rbp-E0h]
  __int64 v86; // [rsp+20h] [rbp-E0h]
  unsigned int v87; // [rsp+28h] [rbp-D8h]
  __int64 v88; // [rsp+28h] [rbp-D8h]
  __int64 v89; // [rsp+28h] [rbp-D8h]
  __int64 v90; // [rsp+28h] [rbp-D8h]
  __int64 v91; // [rsp+28h] [rbp-D8h]
  __int64 v92; // [rsp+28h] [rbp-D8h]
  __int64 v93; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v94; // [rsp+28h] [rbp-D8h]
  __int64 v95; // [rsp+28h] [rbp-D8h]
  __int64 v96; // [rsp+28h] [rbp-D8h]
  __int64 v97; // [rsp+30h] [rbp-D0h]
  int v99; // [rsp+4Ch] [rbp-B4h] BYREF
  __int64 v100; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+58h] [rbp-A8h] BYREF
  _QWORD v102[2]; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD *v103[4]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v104; // [rsp+90h] [rbp-70h]
  __int64 *v105; // [rsp+A0h] [rbp-60h] BYREF
  __int64 *v106; // [rsp+A8h] [rbp-58h]
  __int16 v107; // [rsp+C0h] [rbp-40h]

  v2 = (__int64)a2;
  v3 = *a2;
  v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v5 = *((_QWORD *)a2 - 8);
  v102[0] = a1;
  v99 = v3 - 29;
  v102[1] = &v99;
  v6 = sub_10C7720(v102, v5, (__int64)v4);
  if ( v6 )
    return (__int64)v6;
  v6 = sub_10C7720(v102, (__int64)v4, v5);
  if ( v6 || (unsigned __int8)(*(_BYTE *)v5 - 67) > 0xCu )
    return (__int64)v6;
  v8 = *(_QWORD *)(v5 - 32);
  v9 = *(_QWORD *)(v8 + 8);
  v10 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
    LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
  if ( (_BYTE)v10 != 12 )
    return (__int64)v6;
  v11 = (_BYTE *)*((_QWORD *)a2 - 4);
  v12 = *((_QWORD *)a2 + 1);
  v97 = v12;
  if ( *v11 > 0x15u )
    goto LABEL_10;
  v12 = (unsigned int)*a2 - 29;
  v13 = *(_QWORD *)(v5 + 16);
  v87 = *a2 - 29;
  if ( !v13 )
    goto LABEL_10;
  if ( !*(_QWORD *)(v13 + 8) && *(_BYTE *)v5 == 68 )
  {
    v75 = *(_QWORD *)(v8 + 8);
    v81 = *((_QWORD *)a2 - 4);
    v71 = sub_AD4C30((unsigned __int64)v11, (__int64 **)v9, 0);
    v32 = sub_96F480(0x27u, v71, *(_QWORD *)(v81 + 8), *(_QWORD *)(a1 + 88));
    v11 = (_BYTE *)v81;
    v9 = v75;
    if ( v81 == v32 && v32 != 0 )
    {
      v33 = v71;
      if ( v71 )
      {
        v34 = v71;
        v72 = v75;
        v104 = 257;
        v76 = v33;
        v69 = *(__int64 **)(a1 + 32);
        v35 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v69[10] + 16LL))(
                v69[10],
                v87,
                v8,
                v34);
        v36 = v72;
        v82 = v35;
        if ( !v35 )
        {
          v107 = 257;
          v82 = sub_B504D0(v87, v8, v76, (__int64)&v105, 0, 0);
          v50 = sub_920620(v82);
          v51 = v72;
          if ( v50 )
          {
            v52 = v69[12];
            v53 = *((_DWORD *)v69 + 26);
            if ( v52 )
            {
              sub_B99FD0(v82, 3u, v52);
              v51 = v72;
            }
            v95 = v51;
            sub_B45150(v82, v53);
            v51 = v95;
          }
          v96 = v51;
          (*(void (__fastcall **)(__int64, __int64, _QWORD **, __int64, __int64))(*(_QWORD *)v69[11] + 16LL))(
            v69[11],
            v82,
            v103,
            v69[7],
            v69[8]);
          v36 = v96;
          v54 = *v69 + 16LL * *((unsigned int *)v69 + 2);
          if ( *v69 != v54 )
          {
            v55 = *v69;
            v57 = v54;
            do
            {
              v58 = *(_QWORD *)(v55 + 8);
              v59 = *(_DWORD *)v55;
              v55 += 16;
              sub_B99FD0(v82, v59, v58);
            }
            while ( v57 != v55 );
            v36 = v96;
            v2 = (__int64)a2;
            v6 = 0;
          }
        }
        v90 = v36;
        v107 = 257;
        v37 = sub_BD2C40(72, unk_3F10A14);
        v9 = v90;
        v31 = v37;
        if ( !v37 )
          goto LABEL_10;
        sub_B515B0((__int64)v37, v82, v97, (__int64)&v105, 0, 0);
        v9 = v90;
        goto LABEL_28;
      }
    }
    v13 = *(_QWORD *)(v5 + 16);
    if ( !v13 )
      goto LABEL_10;
  }
  if ( *(_QWORD *)(v13 + 8) )
    goto LABEL_10;
  if ( *(_BYTE *)v5 == 69 )
  {
    v70 = *(_QWORD *)(v5 - 32);
    if ( !v70 )
      goto LABEL_10;
  }
  else
  {
    v77 = v11;
    if ( *(_BYTE *)v5 != 68 )
      goto LABEL_10;
    v85 = v9;
    v60 = sub_B44910(v5);
    v9 = v85;
    if ( !v60 )
      goto LABEL_10;
    v11 = v77;
    v70 = *(_QWORD *)(v5 - 32);
    if ( !v70 )
      goto LABEL_10;
  }
  v73 = v9;
  v78 = v11;
  v25 = sub_AD4C30((unsigned __int64)v11, (__int64 **)v9, 0);
  v26 = sub_96F480(0x28u, v25, *((_QWORD *)v78 + 1), *(_QWORD *)(a1 + 88));
  v9 = v73;
  if ( v78 == (_BYTE *)v26 && v26 != 0 && v25 )
  {
    v79 = v73;
    v104 = 257;
    v74 = *(__int64 **)(a1 + 32);
    v27 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v74[10] + 16LL))(
            v74[10],
            v87,
            v70,
            v25);
    v28 = v79;
    v29 = v27;
    if ( !v27 )
    {
      v107 = 257;
      v91 = sub_B504D0(v87, v70, v25, (__int64)&v105, 0, 0);
      v38 = sub_920620(v91);
      v39 = v91;
      v40 = v79;
      if ( v38 )
      {
        v41 = v74[12];
        v42 = *((_DWORD *)v74 + 26);
        if ( v41 )
        {
          sub_B99FD0(v91, 3u, v41);
          v40 = v79;
          v39 = v91;
        }
        v83 = v40;
        v92 = v39;
        sub_B45150(v39, v42);
        v40 = v83;
        v39 = v92;
      }
      v84 = v40;
      v93 = v39;
      (*(void (__fastcall **)(__int64, __int64, _QWORD **, __int64, __int64))(*(_QWORD *)v74[11] + 16LL))(
        v74[11],
        v39,
        v103,
        v74[7],
        v74[8]);
      v29 = v93;
      v28 = v84;
      v43 = *v74 + 16LL * *((unsigned int *)v74 + 2);
      if ( *v74 != v43 )
      {
        v94 = v4;
        v45 = *v74;
        v46 = v43;
        v47 = v29;
        do
        {
          v48 = *(_QWORD *)(v45 + 8);
          v49 = *(_DWORD *)v45;
          v45 += 16;
          sub_B99FD0(v47, v49, v48);
        }
        while ( v46 != v45 );
        v29 = v47;
        v4 = v94;
        v2 = (__int64)a2;
        v28 = v84;
        v6 = 0;
      }
    }
    v89 = v28;
    v80 = v29;
    v107 = 257;
    v30 = sub_BD2C40(72, unk_3F10A14);
    v9 = v89;
    v31 = v30;
    if ( v30 )
    {
      sub_B51650((__int64)v30, v80, v97, (__int64)&v105, 0, 0);
      v9 = v89;
LABEL_28:
      if ( v31 )
        return (__int64)v31;
    }
  }
LABEL_10:
  v14 = *v4;
  v15 = (unsigned int)(v14 - 67);
  if ( (unsigned __int8)(v14 - 67) <= 0xCu && (_BYTE)v14 == *(_BYTE *)v5 )
  {
    v16 = v14 - 29;
    v88 = *((_QWORD *)v4 - 4);
    if ( v9 == *(_QWORD *)(v88 + 8) )
    {
      v61 = *(_QWORD *)(v5 + 16);
      if ( v61 && !*(_QWORD *)(v61 + 8) || (v62 = *((_QWORD *)v4 + 2)) != 0 && !*(_QWORD *)(v62 + 8) )
      {
        v86 = *(_QWORD *)(v5 - 32);
        if ( (unsigned __int8)sub_10BF400(a1, v5, v15, v12) )
        {
          if ( (unsigned __int8)sub_10BF400(a1, (__int64)v4, v63, v64) )
          {
            v65 = *(__int64 **)(a1 + 32);
            v66 = sub_BD5D20(v2);
            v106 = v67;
            v107 = 261;
            v105 = (__int64 *)v66;
            v68 = sub_10BBE20(v65, v99, v86, v88, (int)v103[0], 0, (__int64)&v105, 0);
            v107 = 257;
            v24 = v68;
            return sub_B51D30(v16, v24, v97, (__int64)&v105, 0, 0);
          }
        }
      }
    }
    else
    {
      v103[0] = &v100;
      v103[1] = &v100;
      if ( (unsigned __int8)sub_10C38C0(v103, v5) )
      {
        v105 = &v101;
        v106 = &v101;
        if ( (unsigned __int8)sub_10C38C0(&v105, (__int64)v4) )
        {
          v17 = v100;
          v18 = sub_BCB060(*(_QWORD *)(v100 + 8));
          v19 = v101;
          v20 = v18;
          v21 = sub_BCB060(*(_QWORD *)(v101 + 8));
          v107 = 257;
          v22 = *(__int64 **)(a1 + 32);
          if ( v20 >= v21 )
            v101 = sub_10BBCE0(v22, v16, v19, *(_QWORD *)(v17 + 8), (__int64)&v105, 0, (int)v103[0], 0);
          else
            v100 = sub_10BBCE0(v22, v16, v17, *(_QWORD *)(v19 + 8), (__int64)&v105, 0, (int)v103[0], 0);
          v107 = 257;
          v23 = sub_10BBE20(*(__int64 **)(a1 + 32), v99, v100, v101, (int)v103[0], 0, (__int64)&v105, 0);
          v107 = 257;
          v24 = v23;
          return sub_B51D30(v16, v24, v97, (__int64)&v105, 0, 0);
        }
      }
    }
  }
  return (__int64)v6;
}
