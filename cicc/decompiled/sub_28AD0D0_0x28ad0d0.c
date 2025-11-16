// Function: sub_28AD0D0
// Address: 0x28ad0d0
//
__int64 __fastcall sub_28AD0D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  unsigned int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r15
  __int64 *v9; // rdi
  _QWORD *v10; // r14
  _QWORD *v11; // r15
  int v12; // esi
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // r8
  int v16; // ecx
  unsigned int v17; // edx
  _QWORD *v18; // rax
  _QWORD *v19; // r11
  __int64 *v20; // r12
  __int64 v21; // r14
  __int64 v22; // rbx
  _BYTE *v23; // r12
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rsi
  __int64 v33; // rcx
  int v34; // edx
  __int64 *v35; // rax
  char *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // eax
  char *v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rax
  char *v43; // rdx
  _BYTE *v44; // r15
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rbx
  _BYTE **v49; // rdx
  _BYTE **v50; // r8
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 *v54; // rbx
  __int64 v55; // rsi
  _QWORD *v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  _QWORD **v60; // rax
  __int64 v61; // rdx
  _QWORD **v62; // rbx
  _QWORD **v63; // r14
  _QWORD *v64; // rsi
  _BYTE *v65; // rbx
  unsigned __int64 v66; // rdi
  unsigned int v67; // ebx
  unsigned int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdi
  int v72; // eax
  unsigned __int8 *v73; // rax
  char *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  _QWORD *v77; // rdx
  unsigned int v78; // eax
  unsigned __int8 *v79; // rax
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  __int64 v82; // rax
  _QWORD *v83; // r15
  unsigned int v84; // eax
  unsigned __int8 *v85; // rax
  int v86; // r9d
  __int64 v87; // [rsp+0h] [rbp-6B0h]
  __int64 v88; // [rsp+0h] [rbp-6B0h]
  int v89; // [rsp+8h] [rbp-6A8h]
  __int64 v90; // [rsp+18h] [rbp-698h]
  unsigned __int8 *v91; // [rsp+18h] [rbp-698h]
  __int64 v92; // [rsp+18h] [rbp-698h]
  unsigned int v93; // [rsp+18h] [rbp-698h]
  __int64 v95; // [rsp+20h] [rbp-690h]
  unsigned int v96; // [rsp+20h] [rbp-690h]
  _BYTE *v98; // [rsp+28h] [rbp-688h]
  unsigned int v99; // [rsp+28h] [rbp-688h]
  __int64 v101; // [rsp+38h] [rbp-678h]
  __int64 v102[2]; // [rsp+40h] [rbp-670h] BYREF
  char *v103; // [rsp+50h] [rbp-660h] BYREF
  __int64 v104; // [rsp+58h] [rbp-658h]
  char v105; // [rsp+60h] [rbp-650h] BYREF
  _QWORD *v106; // [rsp+98h] [rbp-618h]
  void *v107; // [rsp+D0h] [rbp-5E0h]
  _BYTE *v108; // [rsp+E0h] [rbp-5D0h] BYREF
  __int64 v109; // [rsp+E8h] [rbp-5C8h]
  _BYTE v110[1408]; // [rsp+F0h] [rbp-5C0h] BYREF
  __int64 v111; // [rsp+670h] [rbp-40h]

  v101 = sub_B43CC0(a2);
  if ( *(_BYTE *)a2 == 62 )
  {
    v6 = sub_9208B0(v101, *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL));
    v109 = v7;
    v108 = (_BYTE *)v6;
    if ( (_BYTE)v7 )
      return 0;
  }
  v8 = *(_QWORD **)(a2 + 32);
  v9 = 0;
  v108 = v110;
  v10 = v8;
  v109 = 0x800000000LL;
  v111 = v101;
  while ( 1 )
  {
    if ( !v10 )
      BUG();
    v11 = v10 - 3;
    v12 = *((unsigned __int8 *)v10 - 24);
    if ( (unsigned int)(v12 - 30) <= 0xA )
      break;
    v13 = *(_QWORD *)(a1 + 40);
    v14 = *(_DWORD *)(v13 + 56);
    v15 = *(_QWORD *)(v13 + 40);
    if ( v14 )
    {
      v16 = v14 - 1;
      v17 = v16 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v18 = (_QWORD *)(v15 + 16LL * v17);
      v19 = (_QWORD *)*v18;
      if ( v11 == (_QWORD *)*v18 )
      {
LABEL_8:
        v20 = (__int64 *)v18[1];
        if ( !v20 )
          v20 = v9;
        goto LABEL_10;
      }
      v39 = 1;
      while ( v19 != (_QWORD *)-4096LL )
      {
        v86 = v39 + 1;
        v17 = v16 & (v39 + v17);
        v18 = (_QWORD *)(v15 + 16LL * v17);
        v19 = (_QWORD *)*v18;
        if ( v11 == (_QWORD *)*v18 )
          goto LABEL_8;
        v39 = v86;
      }
    }
    v20 = v9;
LABEL_10:
    if ( (unsigned __int8)(v12 - 34) > 0x33u )
      goto LABEL_11;
    v30 = 0x8000000000041LL;
    if ( _bittest64(&v30, (unsigned int)(v12 - 34)) )
    {
      if ( sub_B49EA0((__int64)(v10 - 3)) )
        goto LABEL_23;
      LOBYTE(v12) = *((_BYTE *)v10 - 24);
    }
    if ( (_BYTE)v12 == 62 )
    {
      if ( sub_B46500((unsigned __int8 *)v10 - 24) || (*((_BYTE *)v10 - 22) & 1) != 0 )
        goto LABEL_13;
      v31 = *(v10 - 11);
      v32 = *(_QWORD *)(v31 + 8);
      v33 = v32;
      v34 = *(unsigned __int8 *)(v32 + 8);
      if ( (unsigned int)(v34 - 17) <= 1 )
      {
        v35 = *(__int64 **)(v32 + 16);
        v33 = *v35;
        LOBYTE(v34) = *(_BYTE *)(*v35 + 8);
      }
      if ( (_BYTE)v34 == 14 )
      {
        v90 = *(v10 - 11);
        if ( *((_BYTE *)sub_AE2980(v101, *(_DWORD *)(v33 + 8) >> 8) + 16) )
          goto LABEL_13;
        v31 = v90;
        v32 = *(_QWORD *)(v90 + 8);
      }
      v91 = (unsigned __int8 *)v31;
      v36 = (char *)sub_9208B0(v101, v32);
      v104 = v37;
      v103 = v36;
      if ( (_BYTE)v37 )
        goto LABEL_13;
      v38 = sub_98A180(v91, v101);
      if ( (unsigned int)*a4 - 12 > 1 )
      {
        if ( (unsigned __int8 *)v38 != a4 )
          goto LABEL_13;
      }
      else
      {
        if ( !v38 )
          goto LABEL_13;
        a4 = (unsigned __int8 *)v38;
      }
      v25 = sub_BD58A0(*(v10 - 7), a3, v101);
      v102[1] = v26;
      v102[0] = v25;
      if ( !(_BYTE)v26 )
        goto LABEL_13;
      v27 = sub_9208B0(v111, *(_QWORD *)(*(v10 - 11) + 8LL));
      v104 = v28;
      v103 = (char *)v27;
      v29 = (unsigned __int64)(v27 + 7) >> 3;
      _BitScanReverse64((unsigned __int64 *)&v27, 1LL << (*((_WORD *)v10 - 11) >> 1));
      LOBYTE(v4) = 63 - (v27 ^ 0x3F);
      BYTE1(v4) = 1;
      sub_28A9DA0((__int64)&v108, v102[0], v29, *(v10 - 7), v4, (__int64)(v10 - 3));
LABEL_23:
      v10 = (_QWORD *)v10[1];
      v9 = v20;
    }
    else
    {
      if ( (_BYTE)v12 != 85
        || (v69 = *(v10 - 7)) == 0
        || *(_BYTE *)v69
        || *(_QWORD *)(v69 + 24) != v10[7]
        || (*(_BYTE *)(v69 + 33) & 0x20) == 0
        || ((*(_DWORD *)(v69 + 36) - 243) & 0xFFFFFFFD) != 0 )
      {
LABEL_11:
        if ( (unsigned __int8)sub_B46490((__int64)(v10 - 3)) || (unsigned __int8)sub_B46420((__int64)(v10 - 3)) )
          goto LABEL_13;
        goto LABEL_23;
      }
      v70 = *((_DWORD *)v10 - 5) & 0x7FFFFFF;
      v71 = v11[4 * (3 - v70)];
      if ( *(_DWORD *)(v71 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v71 + 24) )
          goto LABEL_13;
      }
      else
      {
        v89 = *(_DWORD *)(v71 + 32);
        v92 = *((_DWORD *)v10 - 5) & 0x7FFFFFF;
        v72 = sub_C444A0(v71 + 24);
        v70 = v92;
        if ( v89 != v72 )
          goto LABEL_13;
      }
      if ( (unsigned __int8 *)v11[4 * (1 - v70)] != a4
        || *(_BYTE *)v11[4 * (2 - v70)] != 17
        || (v73 = sub_BD3990((unsigned __int8 *)v11[-4 * v70], (__int64)a4),
            v74 = (char *)sub_BD58A0((__int64)v73, a3, v101),
            v104 = v75,
            v103 = v74,
            !(_BYTE)v75) )
      {
LABEL_13:
        v21 = a2;
        v22 = (__int64)v11;
        goto LABEL_14;
      }
      v76 = v11[4 * (2LL - (*((_DWORD *)v10 - 5) & 0x7FFFFFF))];
      v77 = *(_QWORD **)(v76 + 24);
      if ( *(_DWORD *)(v76 + 32) > 0x40u )
        v77 = (_QWORD *)*v77;
      v88 = (__int64)v77;
      LOWORD(v78) = sub_A74840(v10 + 6, 0);
      v93 = v78;
      v79 = sub_BD3990((unsigned __int8 *)v11[-4 * (*((_DWORD *)v10 - 5) & 0x7FFFFFF)], 0);
      sub_28A9DA0((__int64)&v108, (__int64)v103, v88, (__int64)v79, v93, (__int64)(v10 - 3));
      v10 = (_QWORD *)v10[1];
      v9 = v20;
    }
  }
  v21 = a2;
  v22 = (__int64)v11;
  v20 = v9;
LABEL_14:
  if ( !(_DWORD)v109 )
  {
    v95 = 0;
    goto LABEL_16;
  }
  if ( *(_BYTE *)v21 == 62 )
  {
    v40 = (char *)sub_9208B0(v111, *(_QWORD *)(*(_QWORD *)(v21 - 64) + 8LL));
    v42 = v41;
    v103 = v40;
    v43 = v40;
    LOWORD(v40) = *(_WORD *)(v21 + 2);
    v104 = v42;
    _BitScanReverse64((unsigned __int64 *)&v42, 1LL << ((unsigned __int16)v40 >> 1));
    LOBYTE(v40) = 63 - (v42 ^ 0x3F);
    LODWORD(v42) = 256;
    LOBYTE(v42) = (_BYTE)v40;
    sub_28A9DA0((__int64)&v108, 0, (unsigned __int64)(v43 + 7) >> 3, *(_QWORD *)(v21 - 32), (unsigned int)v42, v21);
  }
  else
  {
    v82 = *(_QWORD *)(v21 + 32 * (2LL - (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)));
    v83 = *(_QWORD **)(v82 + 24);
    if ( *(_DWORD *)(v82 + 32) > 0x40u )
      v83 = (_QWORD *)*v83;
    LOWORD(v84) = sub_A74840((_QWORD *)(v21 + 72), 0);
    v99 = v84;
    v85 = sub_BD3990(*(unsigned __int8 **)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)), 0);
    sub_28A9DA0((__int64)&v108, 0, (__int64)v83, (__int64)v85, v99, v21);
  }
  sub_23D0AB0((__int64)&v103, v22, 0, 0, 0);
  v98 = &v108[176 * (unsigned int)v109];
  if ( v98 != v108 )
  {
    v87 = v22;
    v44 = v108;
    v95 = 0;
    while ( 1 )
    {
      v45 = *((_DWORD *)v44 + 10);
      if ( v45 != 1 )
      {
        v46 = *((_QWORD *)v44 + 1);
        v47 = v45;
        v48 = v46 - *(_QWORD *)v44;
        if ( v45 > 3 || v48 > 15 )
        {
LABEL_53:
          v51 = *((_QWORD *)v44 + 2);
          v96 = *((unsigned __int16 *)v44 + 12);
          v52 = sub_BCB2E0(v106);
          v53 = sub_ACD640(v52, v48, 0);
          v95 = sub_B34240((__int64)&v103, v51, (__int64)a4, v53, v96, 0, 0, 0, 0);
          sub_AE9860(v95, *((_QWORD *)v44 + 4), *((unsigned int *)v44 + 10));
          if ( *((_DWORD *)v44 + 10) )
          {
            v54 = (__int64 *)(v95 + 48);
            v55 = *(_QWORD *)(**((_QWORD **)v44 + 4) + 48LL);
            v102[0] = v55;
            if ( v55 )
            {
              sub_B96E90((__int64)v102, v55, 1);
              if ( v54 == v102 )
              {
                if ( v102[0] )
                  sub_B91220((__int64)v102, v102[0]);
                goto LABEL_58;
              }
              v80 = *(_QWORD *)(v95 + 48);
              if ( !v80 )
              {
LABEL_92:
                v81 = (unsigned __int8 *)v102[0];
                *(_QWORD *)(v95 + 48) = v102[0];
                if ( v81 )
                  sub_B976B0((__int64)v102, v81, v95 + 48);
                goto LABEL_58;
              }
LABEL_91:
              sub_B91220(v95 + 48, v80);
              goto LABEL_92;
            }
            if ( v54 != v102 )
            {
              v80 = *(_QWORD *)(v95 + 48);
              if ( v80 )
                goto LABEL_91;
            }
          }
LABEL_58:
          v56 = *(_QWORD **)(a1 + 48);
          if ( v87 == v20[9] )
            v20 = (__int64 *)sub_D69520(v56, v95, 0, (__int64)v20);
          else
            v20 = (__int64 *)sub_D69570(v56, v95, 0, (__int64)v20);
          sub_D75120(*(__int64 **)(a1 + 48), v20, 1);
          v60 = (_QWORD **)*((_QWORD *)v44 + 4);
          v61 = *((unsigned int *)v44 + 10);
          v62 = &v60[v61];
          if ( v62 != v60 )
          {
            v63 = (_QWORD **)*((_QWORD *)v44 + 4);
            do
            {
              v64 = *v63++;
              sub_28AAD10(a1, v64, v61, v57, v58, v59);
            }
            while ( v62 != v63 );
          }
          goto LABEL_63;
        }
        if ( v45 > 1uLL )
        {
          v49 = (_BYTE **)*((_QWORD *)v44 + 4);
          v50 = &v49[v47];
          do
          {
            if ( **v49 != 62 )
              goto LABEL_53;
            ++v49;
          }
          while ( v50 != v49 );
          if ( v47 == 2 )
            goto LABEL_63;
          v67 = v46 - *(_QWORD *)v44;
          v68 = sub_AE44F0(v101);
          if ( v68 <= 7 )
          {
            if ( v67 >= *((_DWORD *)v44 + 10) )
              goto LABEL_63;
LABEL_74:
            v48 = *((_QWORD *)v44 + 1) - *(_QWORD *)v44;
            goto LABEL_53;
          }
          if ( v67 % (v68 >> 3) + v67 / (v68 >> 3) < *((_DWORD *)v44 + 10) )
            goto LABEL_74;
        }
      }
LABEL_63:
      v44 += 176;
      if ( v98 == v44 )
        goto LABEL_64;
    }
  }
  v95 = 0;
LABEL_64:
  nullsub_61();
  v107 = &unk_49DA100;
  nullsub_63();
  if ( v103 != &v105 )
    _libc_free((unsigned __int64)v103);
  v23 = v108;
  v65 = &v108[176 * (unsigned int)v109];
  if ( v108 != v65 )
  {
    do
    {
      v65 -= 176;
      v66 = *((_QWORD *)v65 + 4);
      if ( (_BYTE *)v66 != v65 + 48 )
        _libc_free(v66);
    }
    while ( v23 != v65 );
LABEL_16:
    v23 = v108;
  }
  if ( v23 != v110 )
    _libc_free((unsigned __int64)v23);
  return v95;
}
