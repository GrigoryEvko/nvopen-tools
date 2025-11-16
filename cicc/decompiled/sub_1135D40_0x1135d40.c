// Function: sub_1135D40
// Address: 0x1135d40
//
__int64 __fastcall sub_1135D40(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 **v14; // rax
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // esi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r13
  unsigned int **v35; // rdi
  __int64 v36; // r12
  __int64 v37; // r9
  __int64 v38; // r15
  unsigned __int64 v39; // rsi
  _QWORD *v40; // rax
  int v41; // ecx
  _QWORD *v42; // rdx
  __int64 **v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 *v48; // rax
  __int64 v49; // rax
  unsigned int **v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // r13
  __int64 *v54; // rax
  __int64 v55; // r13
  __int64 v56; // rbx
  __int64 i; // rbx
  __int64 v58; // rdx
  unsigned int v59; // esi
  unsigned __int64 v60; // rsi
  __int64 v61; // [rsp+0h] [rbp-140h]
  bool v62; // [rsp+10h] [rbp-130h]
  __int64 *v63; // [rsp+10h] [rbp-130h]
  __int64 v64; // [rsp+10h] [rbp-130h]
  __int64 v65; // [rsp+10h] [rbp-130h]
  char v66; // [rsp+1Fh] [rbp-121h]
  __int64 v67; // [rsp+20h] [rbp-120h]
  unsigned __int64 v68; // [rsp+28h] [rbp-118h]
  __int64 v69; // [rsp+40h] [rbp-100h] BYREF
  __int64 v70; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE *v71; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v72; // [rsp+58h] [rbp-E8h] BYREF
  int v73; // [rsp+60h] [rbp-E0h] BYREF
  char v74; // [rsp+64h] [rbp-DCh]
  __int64 v75; // [rsp+68h] [rbp-D8h]
  _QWORD v76[4]; // [rsp+70h] [rbp-D0h] BYREF
  char v77; // [rsp+90h] [rbp-B0h]
  char v78; // [rsp+91h] [rbp-AFh]
  _QWORD v79[4]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v80; // [rsp+C0h] [rbp-80h]
  int *v81; // [rsp+D0h] [rbp-70h]
  _QWORD *v82; // [rsp+D8h] [rbp-68h] BYREF
  __int64 *v83; // [rsp+E0h] [rbp-60h]
  _QWORD *v84; // [rsp+E8h] [rbp-58h]
  _QWORD *v85; // [rsp+F0h] [rbp-50h]
  __int16 v86; // [rsp+F8h] [rbp-48h]
  __int64 v87[8]; // [rsp+100h] [rbp-40h] BYREF

  v3 = *(_WORD *)(a2 + 2);
  v73 = 42;
  v74 = 0;
  if ( (v3 & 0x3Fu) - 32 <= 1 )
    goto LABEL_2;
  v10 = *(_QWORD *)(a2 - 64);
  v82 = 0;
  v81 = &v73;
  v83 = &v69;
  v84 = &v71;
  v85 = &v70;
  v11 = *(_QWORD *)(v10 + 16);
  if ( v11
    && !*(_QWORD *)(v11 + 8)
    && *(_BYTE *)v10 == 48
    && (unsigned __int8)sub_995B10(&v82, *(_QWORD *)(v10 - 64))
    && (v23 = *(_QWORD *)(v10 - 32)) != 0
    && (*v83 = v23, *(_BYTE *)v10 > 0x1Cu) )
  {
    *v84 = v10;
    v12 = *(_QWORD *)(a2 - 32);
    if ( v12 )
    {
      *v85 = v12;
      if ( v81 )
      {
        v24 = sub_B53900(a2);
        v25 = (__int64)v81;
        *v81 = v24;
        *(_BYTE *)(v25 + 4) = BYTE4(v24);
      }
      goto LABEL_43;
    }
  }
  else
  {
    v12 = *(_QWORD *)(a2 - 32);
  }
  v13 = *(_QWORD *)(v12 + 16);
  if ( v13 )
  {
    if ( !*(_QWORD *)(v13 + 8) && *(_BYTE *)v12 == 48 )
    {
      if ( (unsigned __int8)sub_995B10(&v82, *(_QWORD *)(v12 - 64)) )
      {
        v19 = *(_QWORD *)(v12 - 32);
        if ( v19 )
        {
          *v83 = v19;
          if ( *(_BYTE *)v12 > 0x1Cu )
          {
            *v84 = v12;
            v20 = *(_QWORD *)(a2 - 64);
            if ( v20 )
            {
              *v85 = v20;
              if ( v81 )
              {
                v21 = sub_B53960(a2);
                v22 = (__int64)v81;
                *v81 = v21;
                *(_BYTE *)(v22 + 4) = BYTE4(v21);
              }
LABEL_43:
              if ( v73 == 35 )
              {
                v62 = 1;
                v15 = 0;
                goto LABEL_46;
              }
              if ( v73 == 36 )
              {
                v62 = 0;
                v15 = 0;
                goto LABEL_46;
              }
              return 0;
            }
          }
        }
      }
    }
  }
  if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
    return 0;
LABEL_2:
  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v4 )
  {
    if ( v5 )
    {
      v70 = *(_QWORD *)(a2 - 32);
      BUG();
    }
    return 0;
  }
  v6 = *(_QWORD *)(v5 + 16);
  v70 = *(_QWORD *)(a2 - 64);
  if ( !v6 )
    goto LABEL_4;
  if ( *(_QWORD *)(v6 + 8) )
    goto LABEL_4;
  if ( (unsigned __int8)(*(_BYTE *)v5 - 48) > 1u )
    goto LABEL_4;
  v14 = (__int64 **)sub_986520(v5);
  v15 = *v14;
  if ( *(_BYTE *)*v14 != 46 )
    goto LABEL_4;
  v16 = *(v15 - 8);
  v17 = *(v15 - 4);
  if ( v4 == v16 && v16 )
  {
    if ( !v17 )
    {
LABEL_4:
      v7 = *(_QWORD *)(v4 + 16);
      v70 = v5;
      if ( !v7 )
        return 0;
      if ( *(_QWORD *)(v7 + 8) )
        return 0;
      if ( (unsigned __int8)(*(_BYTE *)v4 - 48) > 1u )
        return 0;
      v43 = (__int64 **)sub_986520(v4);
      v15 = *v43;
      if ( *(_BYTE *)*v43 != 46 )
        return 0;
      v44 = *(v15 - 8);
      v45 = *(v15 - 4);
      if ( v5 == v44 && v44 )
      {
        if ( !v45 )
          return 0;
        v69 = *(v15 - 4);
      }
      else
      {
        if ( v5 != v45 || v45 == 0 || !v44 )
          return 0;
        v69 = *(v15 - 8);
      }
      if ( *(_BYTE *)v15 > 0x1Cu )
      {
        v46 = sub_986520(v4);
        if ( *(_QWORD *)(v46 + 32) == v69 && *(_BYTE *)v4 > 0x1Cu )
        {
          v71 = (_BYTE *)v4;
          v67 = sub_B53960(a2);
          v73 = v67;
          v74 = BYTE4(v67);
          goto LABEL_24;
        }
      }
      return 0;
    }
    v69 = *(v15 - 4);
  }
  else
  {
    if ( v17 == 0 || v4 != v17 || !v16 )
      goto LABEL_4;
    v69 = *(v15 - 8);
  }
  if ( *(_BYTE *)v15 <= 0x1Cu )
    goto LABEL_4;
  v18 = sub_986520(v5);
  if ( *(_QWORD *)(v18 + 32) != v69 || *(_BYTE *)v5 <= 0x1Cu )
    goto LABEL_4;
  v71 = (_BYTE *)v5;
  v68 = sub_B53900(a2);
  v73 = v68;
  v74 = BYTE4(v68);
LABEL_24:
  v62 = v73 == 32;
LABEL_46:
  v26 = *(_QWORD *)(a1 + 32);
  v81 = (int *)v26;
  v27 = *(_QWORD *)(v26 + 48);
  v82 = 0;
  v83 = 0;
  v84 = (_QWORD *)v27;
  if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
    sub_BD73F0((__int64)&v82);
  v28 = *(_QWORD *)(v26 + 56);
  v86 = *(_WORD *)(v26 + 64);
  v85 = (_QWORD *)v28;
  sub_B33910(v87, (__int64 *)v26);
  v66 = 0;
  if ( v15 )
  {
    v29 = v15[2];
    if ( !v29 || *(_QWORD *)(v29 + 8) )
    {
      sub_D5F1F0(*(_QWORD *)(a1 + 32), (__int64)v15);
      v66 = 1;
    }
  }
  v80 = 259;
  v30 = *(_QWORD *)(a1 + 32);
  v31 = 369;
  v79[0] = "mul";
  v76[1] = v70;
  v76[0] = v69;
  v32 = *(_QWORD *)(v69 + 8);
  BYTE4(v75) = 0;
  v72 = v32;
  if ( *v71 != 48 )
    v31 = 333;
  v33 = sub_B33D10(v30, v31, (__int64)&v72, 1, (int)v76, 2, v75, (__int64)v79);
  v34 = v33;
  if ( v66 )
  {
    v50 = *(unsigned int ***)(a1 + 32);
    v79[0] = "mul.val";
    v80 = 259;
    LODWORD(v76[0]) = 0;
    v51 = sub_94D3D0(v50, v33, (__int64)v76, 1, (__int64)v79);
    sub_F162A0(a1, (__int64)v15, v51);
  }
  v35 = *(unsigned int ***)(a1 + 32);
  v79[0] = "mul.ov";
  v80 = 259;
  LODWORD(v76[0]) = 1;
  v8 = sub_94D3D0(v35, v34, (__int64)v76, 1, (__int64)v79);
  if ( v62 )
  {
    v78 = 1;
    v48 = *(__int64 **)(a1 + 32);
    v76[0] = "mul.not.ov";
    v77 = 3;
    v63 = v48;
    v61 = sub_AD62B0(*(_QWORD *)(v8 + 8));
    v49 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v63[10] + 16LL))(
            v63[10],
            30,
            v8,
            v61);
    if ( !v49 )
    {
      v80 = 257;
      v52 = sub_B504D0(30, v8, v61, (__int64)v79, 0, 0);
      v53 = v63;
      v64 = v52;
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v53[11] + 16LL))(
        v53[11],
        v52,
        v76,
        v53[7],
        v53[8]);
      v54 = v53;
      v55 = *v53;
      v56 = *((unsigned int *)v54 + 2);
      v49 = v64;
      for ( i = v55 + 16 * v56; i != v55; v49 = v65 )
      {
        v58 = *(_QWORD *)(v55 + 8);
        v59 = *(_DWORD *)v55;
        v55 += 16;
        v65 = v49;
        sub_B99FD0(v49, v59, v58);
      }
    }
    v8 = v49;
  }
  if ( v66 )
    sub_F207A0(a1, v15);
  v36 = (__int64)v81;
  if ( v84 )
  {
    sub_A88F30((__int64)v81, (__int64)v84, (__int64)v85, v86);
    v36 = (__int64)v81;
  }
  else
  {
    *((_QWORD *)v81 + 6) = 0;
    *(_QWORD *)(v36 + 56) = 0;
    *(_WORD *)(v36 + 64) = 0;
  }
  v79[0] = v87[0];
  if ( !v87[0] || (sub_B96E90((__int64)v79, v87[0], 1), (v38 = v79[0]) == 0) )
  {
    sub_93FB40(v36, 0);
    v38 = v79[0];
    goto LABEL_87;
  }
  v39 = *(unsigned int *)(v36 + 8);
  v40 = *(_QWORD **)v36;
  v41 = *(_DWORD *)(v36 + 8);
  v42 = (_QWORD *)(*(_QWORD *)v36 + 16 * v39);
  if ( *(_QWORD **)v36 == v42 )
  {
LABEL_90:
    v47 = *(unsigned int *)(v36 + 12);
    if ( v39 >= v47 )
    {
      v60 = v39 + 1;
      if ( v47 < v60 )
      {
        sub_C8D5F0(v36, (const void *)(v36 + 16), v60, 0x10u, v36 + 16, v37);
        v42 = (_QWORD *)(*(_QWORD *)v36 + 16LL * *(unsigned int *)(v36 + 8));
      }
      *v42 = 0;
      v42[1] = v38;
      ++*(_DWORD *)(v36 + 8);
      v38 = v79[0];
    }
    else
    {
      if ( v42 )
      {
        *(_DWORD *)v42 = 0;
        v42[1] = v38;
        v41 = *(_DWORD *)(v36 + 8);
        v38 = v79[0];
      }
      *(_DWORD *)(v36 + 8) = v41 + 1;
    }
LABEL_87:
    if ( !v38 )
      goto LABEL_69;
    goto LABEL_68;
  }
  while ( *(_DWORD *)v40 )
  {
    v40 += 2;
    if ( v42 == v40 )
      goto LABEL_90;
  }
  v40[1] = v79[0];
LABEL_68:
  sub_B91220((__int64)v79, v38);
LABEL_69:
  if ( v87[0] )
    sub_B91220((__int64)v87, v87[0]);
  if ( v84 + 512 != 0 && v84 != 0 && v84 != (_QWORD *)-8192LL )
    sub_BD60C0(&v82);
  return v8;
}
