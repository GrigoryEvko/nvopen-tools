// Function: sub_11E74E0
// Address: 0x11e74e0
//
unsigned __int64 __fastcall sub_11E74E0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // edx
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  int v21; // ebx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  char v26; // al
  __int64 v27; // rdi
  __int64 v28; // r14
  __int64 v29; // r14
  _QWORD *v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 **v33; // r14
  unsigned __int64 v34; // r13
  unsigned int v35; // ebx
  unsigned int v36; // eax
  unsigned __int64 result; // rax
  _BYTE *v38; // rbx
  _BYTE *v39; // rax
  _QWORD **v40; // rbx
  __int64 v41; // r14
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 **v46; // r14
  unsigned __int64 v47; // rbx
  __int64 v48; // rdi
  __int64 (__fastcall *v49)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v50; // r9
  __int64 v51; // rax
  char v52; // bl
  _QWORD *v53; // rax
  __int64 v54; // r14
  unsigned int *v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  _QWORD *v58; // rdi
  __int64 v59; // rax
  _BYTE *v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rax
  __int64 v65; // r14
  __int64 v66; // rax
  char v67; // al
  __int64 v68; // r9
  _QWORD *v69; // r13
  unsigned int *v70; // rbx
  __int64 v71; // r12
  __int64 v72; // rdx
  unsigned int v73; // esi
  unsigned int v74; // eax
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned int *v77; // r14
  __int64 v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  unsigned __int64 v81; // r14
  _BYTE *v82; // rax
  __int64 v83; // rax
  __int64 **v84; // r13
  unsigned int v85; // ebx
  unsigned int v86; // eax
  char v87; // [rsp+8h] [rbp-B8h]
  __int64 *v88; // [rsp+10h] [rbp-B0h]
  __int64 *v90; // [rsp+18h] [rbp-A8h]
  __int64 v91; // [rsp+18h] [rbp-A8h]
  __int64 v92; // [rsp+18h] [rbp-A8h]
  _QWORD **v93; // [rsp+18h] [rbp-A8h]
  __int64 v94; // [rsp+18h] [rbp-A8h]
  __int64 v95; // [rsp+18h] [rbp-A8h]
  void *s; // [rsp+20h] [rbp-A0h] BYREF
  size_t n; // [rsp+28h] [rbp-98h]
  int v98[8]; // [rsp+30h] [rbp-90h] BYREF
  char v99; // [rsp+50h] [rbp-70h]
  char v100; // [rsp+51h] [rbp-6Fh]
  _QWORD v101[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v102; // [rsp+80h] [rbp-40h]

  v5 = *((_DWORD *)a2 + 1);
  s = 0;
  n = 0;
  if ( !(unsigned __int8)sub_98B0F0(*(_QWORD *)&a2[32 * (1LL - (v5 & 0x7FFFFFF))], &s, 1u) )
    return 0;
  v6 = *a2;
  v7 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v6 == 40 )
  {
    v8 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v8 = 0;
    if ( v6 != 85 )
    {
      v8 = 64;
      if ( v6 != 34 )
LABEL_80:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_13;
  v9 = sub_BD2BC0((__int64)a2);
  v11 = v9 + v10;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v11 >> 4) )
LABEL_78:
      BUG();
LABEL_13:
    v15 = 0;
    goto LABEL_14;
  }
  if ( !(unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_13;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_78;
  v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v13 = sub_BD2BC0((__int64)a2);
  v15 = 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
LABEL_14:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v8 - v15) >> 5) == 2 )
  {
    if ( !n || (v38 = s, (v39 = memchr(s, 37, n)) == 0) || v39 - v38 == -1 )
    {
      v90 = *(__int64 **)(a1 + 24);
      v40 = (_QWORD **)sub_B43CA0((__int64)a2);
      v41 = n + 1;
      v42 = sub_97FA80(*v90, (__int64)v40);
      v43 = sub_BCCE00(*v40, v42);
      v44 = sub_ACD640(v43, v41, 0);
      sub_B343C0(
        a3,
        0xEEu,
        v7,
        0x100u,
        *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
        0x100u,
        v44,
        0,
        0,
        0,
        0,
        0);
      return sub_AD64C0(*((_QWORD *)a2 + 1), n, 0);
    }
    return 0;
  }
  if ( n != 2 || *(_BYTE *)s != 37 )
    return 0;
  v16 = *a2;
  if ( v16 == 40 )
  {
    v17 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v17 = 0;
    if ( v16 != 85 )
    {
      v17 = 64;
      if ( v16 != 34 )
        goto LABEL_80;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_42;
  v18 = sub_BD2BC0((__int64)a2);
  v20 = v18 + v19;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v20 >> 4) )
LABEL_77:
      BUG();
LABEL_42:
    v24 = 0;
    goto LABEL_25;
  }
  if ( !(unsigned int)((v20 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_42;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_77;
  v21 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v22 = sub_BD2BC0((__int64)a2);
  v24 = 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v21);
LABEL_25:
  v25 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v25 - 32 - v17 - v24) >> 5) <= 2 )
    return 0;
  v26 = *((_BYTE *)s + 1);
  if ( v26 != 99 )
  {
    if ( v26 == 115 )
    {
      v27 = *(_QWORD *)&a2[32 * (2 - v25)];
      if ( *(_BYTE *)(*(_QWORD *)(v27 + 8) + 8LL) == 14 )
      {
        if ( *((_QWORD *)a2 + 2) )
        {
          v28 = sub_98B430(v27, 8u);
          if ( v28 )
          {
            v88 = *(__int64 **)(a1 + 24);
            v93 = (_QWORD **)sub_B43CA0((__int64)a2);
            v74 = sub_97FA80(*v88, (__int64)v93);
            v75 = sub_BCCE00(*v93, v74);
            v76 = sub_ACD640(v75, v28, 0);
            sub_B343C0(
              a3,
              0xEEu,
              v7,
              0x100u,
              *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
              0x100u,
              v76,
              0,
              0,
              0,
              0,
              0);
            return sub_AD64C0(*((_QWORD *)a2 + 1), v28 - 1, 0);
          }
          v29 = sub_11CA2E0(
                  v7,
                  *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                  a3,
                  *(__int64 **)(a1 + 24));
          if ( v29 )
          {
            v30 = *(_QWORD **)(a3 + 72);
            v102 = 257;
            v31 = (_QWORD *)sub_BCB2B0(v30);
            v32 = sub_B36570((unsigned int **)a3, v31, v29, v7, (__int64)v101);
            v33 = (__int64 **)*((_QWORD *)a2 + 1);
            v102 = 257;
            v34 = v32;
            v35 = sub_BCB060(*(_QWORD *)(v32 + 8));
            v36 = sub_BCB060((__int64)v33);
            return sub_11DB4B0((__int64 *)a3, (unsigned int)(v35 <= v36) + 38, v34, v33, (__int64)v101, 0, v98[0], 0);
          }
          if ( !(unsigned __int8)sub_11F3070(*((_QWORD *)a2 + 5), *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 64), 0) )
          {
            v81 = sub_11CA050(
                    *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                    a3,
                    *(_QWORD *)(a1 + 16),
                    *(__int64 **)(a1 + 24));
            if ( v81 )
            {
              v101[0] = "leninc";
              v102 = 259;
              v82 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v81 + 8), 1, 0);
              v83 = sub_929C50((unsigned int **)a3, (_BYTE *)v81, v82, (__int64)v101, 0, 0);
              sub_B343C0(
                a3,
                0xEEu,
                v7,
                0x100u,
                *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                0x100u,
                v83,
                0,
                0,
                0,
                0,
                0);
              v84 = (__int64 **)*((_QWORD *)a2 + 1);
              v102 = 257;
              v85 = sub_BCB060(*(_QWORD *)(v81 + 8));
              v86 = sub_BCB060((__int64)v84);
              return sub_11DB4B0((__int64 *)a3, (unsigned int)(v85 <= v86) + 38, v81, v84, (__int64)v101, 0, v98[0], 0);
            }
          }
          return 0;
        }
        result = sub_11CA290(v7, *(_QWORD *)&a2[32 * (2 - v25)], a3, *(__int64 **)(a1 + 24));
        if ( result )
        {
          if ( *(_BYTE *)result == 85 )
            *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *((_WORD *)a2 + 1) & 3;
          return result;
        }
      }
    }
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[32 * (2 - v25)] + 8LL) + 8LL) != 12 )
    return 0;
  v45 = *(_QWORD **)(a3 + 72);
  v100 = 1;
  *(_QWORD *)v98 = "char";
  v99 = 3;
  v46 = (__int64 **)sub_BCB2B0(v45);
  v47 = *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  if ( v46 == *(__int64 ***)(v47 + 8) )
  {
    v50 = *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    goto LABEL_52;
  }
  v48 = *(_QWORD *)(a3 + 80);
  v49 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v48 + 120LL);
  if ( v49 == sub_920130 )
  {
    if ( *(_BYTE *)v47 > 0x15u )
      goto LABEL_65;
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v50 = sub_ADAB70(38, v47, v46, 0);
    else
      v50 = sub_AA93C0(0x26u, v47, (__int64)v46);
  }
  else
  {
    v50 = v49(v48, 38u, (_BYTE *)v47, (__int64)v46);
  }
  if ( !v50 )
  {
LABEL_65:
    v102 = 257;
    v94 = sub_B51D30(38, v47, (__int64)v46, (__int64)v101, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v94,
      v98,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v77 = *(unsigned int **)a3;
    v50 = v94;
    v78 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v78 )
    {
      do
      {
        v79 = *((_QWORD *)v77 + 1);
        v80 = *v77;
        v77 += 4;
        v95 = v50;
        sub_B99FD0(v50, v80, v79);
        v50 = v95;
      }
      while ( (unsigned int *)v78 != v77 );
    }
  }
LABEL_52:
  v91 = v50;
  v51 = sub_AA4E30(*(_QWORD *)(a3 + 48));
  v52 = sub_AE5020(v51, *(_QWORD *)(v91 + 8));
  v102 = 257;
  v53 = sub_BD2C40(80, unk_3F10A10);
  v54 = (__int64)v53;
  if ( v53 )
    sub_B4D3C0((__int64)v53, v91, v7, 0, v52, v91, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v54,
    v101,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v55 = *(unsigned int **)a3;
  v92 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v92 )
  {
    do
    {
      v56 = *((_QWORD *)v55 + 1);
      v57 = *v55;
      v55 += 4;
      sub_B99FD0(v54, v57, v56);
    }
    while ( (unsigned int *)v92 != v55 );
  }
  v58 = *(_QWORD **)(a3 + 72);
  v101[0] = "nul";
  v102 = 259;
  v59 = sub_BCB2D0(v58);
  v60 = (_BYTE *)sub_ACD640(v59, 1, 0);
  v61 = *(_QWORD **)(a3 + 72);
  *(_QWORD *)v98 = v60;
  v62 = sub_BCB2B0(v61);
  v63 = sub_921130((unsigned int **)a3, v62, v7, (_BYTE **)v98, 1, (__int64)v101, 3u);
  v64 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
  v65 = sub_ACD640(v64, 0, 0);
  v66 = sub_AA4E30(*(_QWORD *)(a3 + 48));
  v67 = sub_AE5020(v66, *(_QWORD *)(v65 + 8));
  v102 = 257;
  v87 = v67;
  v69 = sub_BD2C40(80, unk_3F10A10);
  if ( v69 )
    sub_B4D3C0((__int64)v69, v65, v63, 0, v87, v68, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v69,
    v101,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v70 = *(unsigned int **)a3;
  v71 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  while ( (unsigned int *)v71 != v70 )
  {
    v72 = *((_QWORD *)v70 + 1);
    v73 = *v70;
    v70 += 4;
    sub_B99FD0((__int64)v69, v73, v72);
  }
  return sub_AD64C0(*((_QWORD *)a2 + 1), 1, 0);
}
