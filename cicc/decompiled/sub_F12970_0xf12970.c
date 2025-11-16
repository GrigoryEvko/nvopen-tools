// Function: sub_F12970
// Address: 0xf12970
//
__int64 __fastcall sub_F12970(unsigned __int8 **a1, int a2)
{
  unsigned __int8 *v3; // rdx
  char *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r8
  unsigned __int8 v8; // r9
  int v9; // ecx
  __int64 *v10; // rdi
  __int64 v11; // r13
  _BYTE *v12; // r12
  unsigned __int8 *v13; // rsi
  unsigned __int8 v14; // al
  __int64 v15; // r11
  __int64 v16; // r10
  __int64 v17; // r14
  __int64 v18; // rsi
  char v19; // si
  __int64 *v20; // r10
  __int64 v21; // r15
  int v22; // r9d
  char v23; // si
  int v24; // r14d
  unsigned int v25; // r11d
  __int64 *v26; // rdi
  __int64 *v27; // r15
  __int64 v28; // rax
  __int64 v29; // r10
  __int64 v30; // r13
  unsigned __int8 *v31; // rax
  __int64 *v32; // r14
  char v33; // al
  __int64 v34; // r10
  char v35; // al
  char v36; // al
  __int64 v37; // r10
  int v38; // r11d
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rcx
  unsigned int v45; // edi
  __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rcx
  __int64 *v49; // rdi
  unsigned __int8 *v50; // rax
  __int64 v51; // rax
  char v52; // al
  __int64 *v53; // rax
  __int64 v54; // r11
  __int64 *v55; // rdi
  unsigned __int8 *v56; // rax
  __int64 v57; // rax
  char v58; // al
  __int64 v59; // r10
  __int64 v60; // rdx
  int v61; // r12d
  __int64 v62; // r12
  __int64 v63; // r15
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // rax
  __int64 *v67; // r15
  __int64 v68; // rdi
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // r15
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // [rsp+8h] [rbp-A8h]
  int v78; // [rsp+8h] [rbp-A8h]
  __int64 v79; // [rsp+10h] [rbp-A0h]
  unsigned int v80; // [rsp+10h] [rbp-A0h]
  __int64 v81; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v82; // [rsp+10h] [rbp-A0h]
  int v83; // [rsp+10h] [rbp-A0h]
  __int64 v84; // [rsp+10h] [rbp-A0h]
  __int64 i; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v86; // [rsp+10h] [rbp-A0h]
  unsigned int v87; // [rsp+18h] [rbp-98h]
  char *v88; // [rsp+18h] [rbp-98h]
  int v89; // [rsp+18h] [rbp-98h]
  __int64 v90; // [rsp+18h] [rbp-98h]
  __int64 *v91; // [rsp+18h] [rbp-98h]
  __int64 v92; // [rsp+18h] [rbp-98h]
  __int64 v93; // [rsp+18h] [rbp-98h]
  __int64 v94; // [rsp+18h] [rbp-98h]
  __int64 v95; // [rsp+18h] [rbp-98h]
  __int64 v96; // [rsp+18h] [rbp-98h]
  __int64 v97; // [rsp+18h] [rbp-98h]
  int v98[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v99; // [rsp+40h] [rbp-70h]
  _QWORD *v100[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v101; // [rsp+70h] [rbp-40h]

  v3 = *a1;
  v4 = *(char **)&(*a1)[32 * a2 - 64];
  v5 = *((_QWORD *)v4 + 2);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(v5 + 8);
  if ( v6 )
    return 0;
  v8 = *v4;
  if ( (unsigned __int8)*v4 <= 0x1Cu )
    return 0;
  v9 = v8;
  if ( (unsigned int)v8 - 54 > 2 )
    return 0;
  if ( (v4[7] & 0x40) != 0 )
  {
    v10 = (__int64 *)*((_QWORD *)v4 - 1);
    v11 = *v10;
    if ( !*v10 )
      return 0;
  }
  else
  {
    v10 = (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
    v11 = *v10;
    if ( !*v10 )
      return 0;
  }
  v12 = (_BYTE *)v10[4];
  if ( !v12 )
    return 0;
  v13 = *(unsigned __int8 **)&v3[32 * (1 - a2) - 64];
  v14 = *v13;
  if ( *v13 <= 0x1Cu || (unsigned int)v14 - 42 > 0x11 )
    return v6;
  v15 = *((_QWORD *)v13 - 8);
  v16 = *((_QWORD *)v13 - 4);
  v17 = *(_QWORD *)(v15 + 16);
  if ( v17 )
  {
    if ( !*(_QWORD *)(v17 + 8) )
    {
      v19 = *(_BYTE *)v15;
      if ( (unsigned __int8)(*(_BYTE *)v15 - 54) <= 2u )
      {
        if ( (*(_BYTE *)(v15 + 7) & 0x40) != 0 )
        {
          v32 = *(__int64 **)(v15 - 8);
          v21 = *v32;
          if ( *v32 )
          {
LABEL_35:
            if ( v12 == (_BYTE *)v32[4] && v16 )
              goto LABEL_20;
          }
        }
        else
        {
          v32 = (__int64 *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
          v21 = *v32;
          if ( *v32 )
            goto LABEL_35;
        }
      }
    }
  }
  v18 = *(_QWORD *)(v16 + 16);
  if ( !v18 )
    return v6;
  if ( *(_QWORD *)(v18 + 8) )
    return v6;
  v19 = *(_BYTE *)v16;
  if ( (unsigned __int8)(*(_BYTE *)v16 - 54) > 2u )
    return v6;
  if ( (*(_BYTE *)(v16 + 7) & 0x40) != 0 )
  {
    v20 = *(__int64 **)(v16 - 8);
    v21 = *v20;
    if ( !*v20 )
      return v6;
  }
  else
  {
    v20 = (__int64 *)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
    v21 = *v20;
    if ( !*v20 )
      return v6;
  }
  if ( v12 != (_BYTE *)v20[4] )
    return v6;
  v16 = v15;
LABEL_20:
  if ( v19 != v8 )
    return v6;
  v22 = *v3;
  v23 = *v3;
  if ( (_BYTE)v22 != 42 && (unsigned __int8)(v22 - 57) > 2u )
    return v6;
  v24 = v14 - 29;
  if ( v14 == 42 )
  {
    v25 = v9 - 29;
    if ( v9 == 56 )
      return v6;
  }
  else
  {
    if ( (unsigned __int8)(v14 - 57) > 2u )
      return v6;
    v25 = v9 - 29;
    if ( v9 == 56 )
    {
      if ( (unsigned int)(v22 - 57) <= 2 && v14 == 59 )
      {
        v100[0] = 0;
        v52 = sub_995B10(v100, v16);
        v6 = 0;
        if ( v52 )
        {
          v53 = (__int64 *)*((_QWORD *)a1[2] + 4);
          v99 = 257;
          v91 = v53;
          v84 = sub_AD62B0(*(_QWORD *)(v21 + 8));
          v54 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v91[10] + 16LL))(
                  v91[10],
                  30,
                  v21,
                  v84);
          if ( !v54 )
          {
            v101 = 257;
            v66 = sub_B504D0(30, v21, v84, (__int64)v100, 0, 0);
            v67 = v91;
            v68 = v91[11];
            v69 = v91[7];
            v95 = v66;
            (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v68 + 16LL))(
              v68,
              v66,
              v98,
              v69,
              v67[8]);
            v54 = v95;
            v70 = *v67 + 16LL * *((unsigned int *)v67 + 2);
            v71 = *v67;
            for ( i = v70; i != v71; v54 = v96 )
            {
              v72 = *(_QWORD *)(v71 + 8);
              v73 = *(_DWORD *)v71;
              v71 += 16;
              v96 = v54;
              sub_B99FD0(v54, v73, v72);
            }
          }
          v55 = (__int64 *)*((_QWORD *)a1[2] + 4);
          v56 = *a1;
          v101 = 257;
          v57 = sub_F0A990(v55, (unsigned int)*v56 - 29, v11, v54, v98[0], 0, (__int64)v100, 0);
          v101 = 257;
          return sub_B504D0(27, v57, (__int64)v12, (__int64)v100, 0, 0);
        }
      }
      return v6;
    }
  }
  if ( v23 == v14 && (v24 != 13 && v23 != 42 || v25 == 25) )
  {
    v79 = v16;
    v26 = (__int64 *)*((_QWORD *)a1[2] + 4);
    v101 = 257;
    v87 = v25;
    v77 = sub_F0A990(v26, (unsigned int)*v3 - 29, v21, v11, v98[0], 0, (__int64)v100, 0);
    v27 = (__int64 *)*((_QWORD *)a1[2] + 4);
    v99 = 257;
    v28 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _BYTE *))(*(_QWORD *)v27[10] + 16LL))(
            v27[10],
            v87,
            v77,
            v12);
    v29 = v79;
    v30 = v28;
    if ( !v28 )
    {
      v101 = 257;
      v30 = sub_B504D0(v87, v77, (__int64)v12, (__int64)v100, 0, 0);
      v58 = sub_920620(v30);
      v59 = v79;
      if ( v58 )
      {
        v60 = v27[12];
        v61 = *((_DWORD *)v27 + 26);
        if ( v60 )
        {
          sub_B99FD0(v30, 3u, v60);
          v59 = v79;
        }
        v92 = v59;
        sub_B45150(v30, v61);
        v59 = v92;
      }
      v93 = v59;
      (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v27[11] + 16LL))(
        v27[11],
        v30,
        v98,
        v27[7],
        v27[8]);
      v29 = v93;
      v62 = *v27 + 16LL * *((unsigned int *)v27 + 2);
      if ( *v27 != v62 )
      {
        v63 = *v27;
        do
        {
          v64 = *(_QWORD *)(v63 + 8);
          v65 = *(_DWORD *)v63;
          v63 += 16;
          v94 = v29;
          sub_B99FD0(v30, v65, v64);
          v29 = v94;
        }
        while ( v62 != v63 );
      }
    }
    v31 = *a1;
    v101 = 257;
    return sub_B504D0((unsigned int)*v31 - 29, v30, v29, (__int64)v100, 0, 0);
  }
  else
  {
    v80 = v25;
    v88 = (char *)v16;
    if ( *v12 <= 0x15u && *v12 != 5 )
    {
      v33 = sub_AD6CA0((__int64)v12);
      v6 = 0;
      if ( !v33 )
      {
        v34 = (__int64)v88;
        v35 = *v88;
        if ( (unsigned __int8)*v88 <= 0x15u )
        {
          v89 = v80;
          if ( v35 != 5 )
          {
            v81 = v34;
            v36 = sub_AD6CA0(v34);
            v6 = 0;
            if ( !v36 )
            {
              v37 = v81;
              v38 = v89;
              v39 = **a1;
              if ( v39 == 57 )
                goto LABEL_61;
              if ( (v24 == 13 || v39 == 42) && v89 != 25 )
                return v6;
              if ( v24 == 28 )
              {
LABEL_61:
                v44 = (__int64)a1[6];
                v45 = 51 - v89;
              }
              else
              {
                v40 = *((_QWORD *)a1[4] + 2);
                if ( v89 == 26 )
                {
                  v74 = v81;
                  v97 = v81;
                  v86 = a1[4];
                  v75 = sub_96E6C0(0x19u, v74, v12, v40);
                  v76 = sub_96E6C0(0x1Au, v75, v12, *((_QWORD *)v86 + 2));
                  v37 = v97;
                  v38 = 26;
                  v6 = 0;
                  if ( v97 != v76 )
                    return v6;
                  v44 = (__int64)a1[6];
                  v45 = 25;
                }
                else
                {
                  v41 = v81;
                  v78 = v89;
                  v90 = v81;
                  v82 = a1[4];
                  v42 = sub_96E6C0(0x1Au, v41, v12, v40);
                  v43 = sub_96E6C0(0x19u, v42, v12, *((_QWORD *)v82 + 2));
                  v37 = v90;
                  v38 = v78;
                  v6 = 0;
                  if ( v90 != v43 )
                    return v6;
                  v44 = (__int64)a1[6];
                  v45 = 26;
                }
              }
              v83 = v38;
              v46 = sub_96E6C0(v45, v37, v12, v44);
              v47 = (__int64 *)*((_QWORD *)a1[2] + 4);
              v101 = 257;
              v48 = sub_F0A990(v47, v24, v21, v46, v98[0], 0, (__int64)v100, 0);
              v49 = (__int64 *)*((_QWORD *)a1[2] + 4);
              v50 = *a1;
              v101 = 257;
              v51 = sub_F0A990(v49, (unsigned int)*v50 - 29, v11, v48, v98[0], 0, (__int64)v100, 0);
              v101 = 257;
              return sub_B504D0(v83, v51, (__int64)v12, (__int64)v100, 0, 0);
            }
          }
        }
      }
    }
  }
  return v6;
}
