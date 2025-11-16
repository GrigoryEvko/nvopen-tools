// Function: sub_327D4E0
// Address: 0x327d4e0
//
__int64 __fastcall sub_327D4E0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 *v4; // rax
  unsigned __int16 *v5; // rdx
  unsigned __int64 v6; // r12
  int v7; // eax
  __int64 v8; // r12
  unsigned __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned int v20; // r9d
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 *v23; // rdi
  unsigned int v24; // r8d
  unsigned __int16 v25; // r12
  __int64 v26; // rdx
  int v27; // r10d
  __int64 v28; // r11
  unsigned int v29; // edx
  __int64 v30; // rcx
  bool (__fastcall *v31)(__int64, __int64, unsigned __int16); // rax
  __int64 (__fastcall *v32)(__int64, __int64, unsigned int); // rax
  __int64 (*v33)(); // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  char v36; // al
  unsigned int v37; // r9d
  unsigned int v38; // r10d
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __int64 v41; // r12
  __int128 v42; // rax
  int v43; // r9d
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // r9
  __int64 v48; // r11
  int v49; // eax
  unsigned __int64 v50; // rax
  __int64 v51; // r9
  __int64 v52; // r11
  int v53; // eax
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  unsigned __int64 v59; // rax
  unsigned int v60; // eax
  unsigned int v61; // eax
  __int16 v62; // [rsp+Ah] [rbp-C6h]
  unsigned int v63; // [rsp+14h] [rbp-BCh]
  __int64 v64; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v65; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v66; // [rsp+18h] [rbp-B8h]
  unsigned int v67; // [rsp+18h] [rbp-B8h]
  __int64 v68; // [rsp+20h] [rbp-B0h]
  __int64 v69; // [rsp+20h] [rbp-B0h]
  __int64 v70; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+20h] [rbp-B0h]
  __int64 v72; // [rsp+20h] [rbp-B0h]
  unsigned int v73; // [rsp+28h] [rbp-A8h]
  unsigned int v74; // [rsp+28h] [rbp-A8h]
  __int64 v75; // [rsp+28h] [rbp-A8h]
  int v76; // [rsp+28h] [rbp-A8h]
  int v77; // [rsp+28h] [rbp-A8h]
  unsigned int v78; // [rsp+30h] [rbp-A0h]
  unsigned int v79; // [rsp+30h] [rbp-A0h]
  __int16 v80; // [rsp+32h] [rbp-9Eh]
  __int16 v81; // [rsp+32h] [rbp-9Eh]
  unsigned __int64 v82; // [rsp+38h] [rbp-98h]
  unsigned int v83; // [rsp+38h] [rbp-98h]
  unsigned int v84; // [rsp+38h] [rbp-98h]
  unsigned int v85; // [rsp+38h] [rbp-98h]
  unsigned __int64 v86; // [rsp+40h] [rbp-90h]
  unsigned __int64 v87; // [rsp+48h] [rbp-88h]
  unsigned int v88; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v89; // [rsp+58h] [rbp-78h]
  __int64 v90; // [rsp+60h] [rbp-70h] BYREF
  int v91; // [rsp+68h] [rbp-68h]
  __int64 v92; // [rsp+70h] [rbp-60h]
  __int64 v93; // [rsp+78h] [rbp-58h]
  unsigned __int64 v94; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int64 v95; // [rsp+88h] [rbp-48h]
  unsigned __int64 v96; // [rsp+90h] [rbp-40h]
  unsigned int v97; // [rsp+98h] [rbp-38h]

  v4 = *(unsigned __int64 **)(a2 + 40);
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *((_QWORD *)v5 + 1);
  v87 = *v4;
  v86 = v4[1];
  v82 = *v4;
  v78 = *((_DWORD *)v4 + 2);
  v7 = *v5;
  v89 = v6;
  LOWORD(v88) = v7;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      LOWORD(v94) = v7;
      v95 = v6;
      goto LABEL_4;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v9 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v88) )
    {
      v95 = v6;
      LOWORD(v94) = 0;
      goto LABEL_9;
    }
    LOWORD(v7) = sub_3009970((__int64)&v88, a2, v17, v18, v19);
  }
  LOWORD(v94) = v7;
  v95 = v9;
  if ( (_WORD)v7 )
  {
LABEL_4:
    if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
      BUG();
    v8 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
    goto LABEL_10;
  }
LABEL_9:
  v92 = sub_3007260((__int64)&v94);
  LODWORD(v8) = v92;
  v93 = v10;
LABEL_10:
  v11 = *(_QWORD *)(a2 + 80);
  v90 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v90, v11, 1);
  v12 = *a1;
  v91 = *(_DWORD *)(a2 + 72);
  v94 = v87;
  v95 = v86;
  v13 = sub_3402EA0(v12, 200, (unsigned int)&v90, v88, v89, 0, (__int64)&v94, 1);
  v14 = v13;
  if ( v13 )
  {
    v15 = v13;
    goto LABEL_14;
  }
  if ( ((*(_DWORD *)(v82 + 24) - 190) & 0xFFFFFFFD) != 0 )
    goto LABEL_20;
  v46 = sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(v82 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v82 + 40) + 48LL), 0, 0);
  if ( !v46 )
    goto LABEL_20;
  v47 = *(_QWORD *)(v46 + 96);
  v48 = v47 + 24;
  v74 = *(_DWORD *)(v47 + 32);
  if ( v74 <= 0x40 )
  {
    v50 = *(_QWORD *)(v47 + 24);
  }
  else
  {
    v64 = *(_QWORD *)(v46 + 96);
    v68 = v47 + 24;
    v49 = sub_C444A0(v47 + 24);
    v48 = v68;
    v47 = v64;
    if ( v74 - v49 > 0x40 )
      goto LABEL_20;
    v50 = **(_QWORD **)(v64 + 24);
  }
  v69 = v48;
  v75 = v47;
  if ( (unsigned int)v8 <= v50 )
  {
LABEL_20:
    if ( (_WORD)v88 )
    {
      if ( (unsigned __int16)(v88 - 2) > 7u )
        goto LABEL_44;
    }
    else if ( !sub_30070A0((__int64)&v88) )
    {
      goto LABEL_44;
    }
    v20 = v8;
    if ( (unsigned int)v8 <= 8 || (v8 & 1) != 0 )
      goto LABEL_44;
    v21 = (unsigned int)v8 >> 1;
    v73 = (unsigned int)v8 >> 1;
    if ( (unsigned int)v8 >> 1 == 4 )
    {
      v25 = 4;
    }
    else
    {
      switch ( v21 )
      {
        case 8u:
          v25 = 5;
          break;
        case 0x10u:
          v25 = 6;
          break;
        case 0x20u:
          v25 = 7;
          break;
        case 0x40u:
          v25 = 8;
          break;
        case 0x80u:
          v25 = 9;
          break;
        default:
          v22 = sub_3007020(*(_QWORD **)(*a1 + 64LL), v21);
          v23 = (__int64 *)a1[1];
          v24 = v22;
          v20 = v8;
          v25 = v22;
          v14 = v26;
          HIWORD(v27) = HIWORD(v22);
          v28 = v26;
          if ( *((_BYTE *)a1 + 33) )
          {
            v29 = 1;
            if ( (_WORD)v22 == 1 )
              goto LABEL_32;
            if ( !(_WORD)v22 )
              goto LABEL_44;
LABEL_96:
            v29 = v25;
            if ( !v23[v25 + 14] )
              goto LABEL_44;
LABEL_32:
            if ( *((_BYTE *)v23 + 500 * v29 + 6614) )
              goto LABEL_44;
            v30 = *v23;
            v31 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*v23 + 2192);
            if ( v31 == sub_302E170 )
              goto LABEL_34;
            goto LABEL_98;
          }
          if ( (_WORD)v22 == 1 )
          {
            v29 = 1;
            goto LABEL_83;
          }
LABEL_81:
          if ( !v25 )
            goto LABEL_44;
          v29 = v25;
          if ( !v23[v25 + 14] )
            goto LABEL_44;
LABEL_83:
          if ( (*((_BYTE *)v23 + 500 * v29 + 6614) & 0xFB) != 0 )
            goto LABEL_44;
          v30 = *v23;
          v31 = *(bool (__fastcall **)(__int64, __int64, unsigned __int16))(*v23 + 2192);
          if ( v31 == sub_302E170 )
          {
            if ( !v25 )
              goto LABEL_44;
LABEL_34:
            if ( !v23[(int)v29 + 14] )
              goto LABEL_44;
            goto LABEL_35;
          }
LABEL_98:
          v62 = HIWORD(v27);
          v63 = v20;
          v67 = v24;
          v72 = v28;
          if ( !((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD, __int64))v31)(v23, 200, v24, v28) )
            goto LABEL_44;
          v23 = (__int64 *)a1[1];
          HIWORD(v27) = v62;
          v20 = v63;
          v24 = v67;
          v30 = *v23;
          v28 = v72;
LABEL_35:
          v32 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v30 + 1408);
          if ( v32 != sub_2FE3A30 )
          {
            v81 = HIWORD(v27);
            v85 = v20;
            v36 = ((__int64 (__fastcall *)(__int64 *, unsigned __int64, unsigned __int64, _QWORD, __int64))v32)(
                    v23,
                    v87,
                    v86,
                    v24,
                    v28);
            HIWORD(v38) = v81;
            v37 = v85;
            goto LABEL_38;
          }
          v33 = *(__int64 (**)())(v30 + 1392);
          v34 = *(_QWORD *)(*(_QWORD *)(v82 + 48) + 16LL * v78 + 8);
          v35 = *(unsigned __int16 *)(*(_QWORD *)(v82 + 48) + 16LL * v78);
          if ( v33 != sub_2FE3480 )
          {
            v80 = HIWORD(v27);
            v83 = v20;
            v36 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, __int64))v33)(v23, v35, v34, v24, v28);
            v37 = v83;
            HIWORD(v38) = v80;
LABEL_38:
            if ( v36 )
            {
              v39 = a1[1];
              v40 = *(__int64 (**)())(*(_QWORD *)v39 + 1432LL);
              if ( v40 != sub_2FE34A0 )
              {
                LOWORD(v38) = v25;
                v84 = v37;
                v79 = v38;
                if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, unsigned __int64))v40)(
                       v39,
                       v38,
                       v14,
                       v88,
                       v89) )
                {
                  sub_109DDE0((__int64)&v94, v84, v73);
                  if ( (unsigned __int8)sub_33DD210(*a1, v87, v86, &v94, 0) )
                  {
                    v41 = *a1;
                    *(_QWORD *)&v42 = sub_33FB310(*a1, v87, v86, &v90, v79, v14);
                    v44 = sub_33FAF80(v41, 200, (unsigned int)&v90, v79, v14, v43, v42);
                    v15 = sub_33FB310(*a1, v44, v45, &v90, v88, v89);
                    sub_969240((__int64 *)&v94);
                    goto LABEL_14;
                  }
                  sub_969240((__int64 *)&v94);
                }
              }
            }
          }
LABEL_44:
          v15 = 0;
          goto LABEL_14;
      }
    }
    v28 = 0;
    v23 = (__int64 *)a1[1];
    v24 = v25;
    v27 = v25;
    if ( *((_BYTE *)a1 + 33) )
      goto LABEL_96;
    goto LABEL_81;
  }
  sub_33DD090(&v94, *a1, **(_QWORD **)(v82 + 40), *(_QWORD *)(*(_QWORD *)(v82 + 40) + 8LL), 0);
  v51 = v75;
  v52 = v69;
  v53 = *(_DWORD *)(v82 + 24);
  if ( v53 != 192 )
  {
    if ( v53 == 190 )
    {
      if ( (unsigned int)v95 > 0x40 )
      {
        v60 = sub_C44500((__int64)&v94);
        v51 = v75;
        v52 = v69;
        v54 = v60;
      }
      else if ( (_DWORD)v95 )
      {
        v54 = 64;
        if ( v94 << (64 - (unsigned __int8)v95) != -1 )
        {
          _BitScanReverse64(&v55, ~(v94 << (64 - (unsigned __int8)v95)));
          v54 = (int)(v55 ^ 0x3F);
        }
      }
      else
      {
        v54 = 0;
      }
      if ( *(_DWORD *)(v51 + 32) <= 0x40u )
      {
        v56 = *(_QWORD *)(v51 + 24);
        goto LABEL_58;
      }
      v65 = v54;
      v70 = v51;
      v76 = *(_DWORD *)(v51 + 32);
      if ( v76 - (unsigned int)sub_C444A0(v52) <= 0x40 )
      {
        LODWORD(v51) = v70;
        v54 = v65;
        v56 = **(_QWORD **)(v70 + 24);
LABEL_58:
        if ( v54 >= v56 )
          goto LABEL_59;
      }
    }
LABEL_73:
    if ( v97 > 0x40 && v96 )
      j_j___libc_free_0_0(v96);
    if ( (unsigned int)v95 > 0x40 && v94 )
      j_j___libc_free_0_0(v94);
    goto LABEL_20;
  }
  if ( (unsigned int)v95 > 0x40 )
  {
    v61 = sub_C445E0((__int64)&v94);
    v52 = v69;
    v51 = v75;
    _RDX = v61;
  }
  else
  {
    _RAX = ~v94;
    __asm { tzcnt   rdx, rax }
    _RDX = (int)_RDX;
    if ( v94 == -1 )
      _RDX = 64;
  }
  if ( *(_DWORD *)(v51 + 32) > 0x40u )
  {
    v66 = _RDX;
    v71 = v51;
    v77 = *(_DWORD *)(v51 + 32);
    if ( v77 - (unsigned int)sub_C444A0(v52) > 0x40 )
      goto LABEL_73;
    LODWORD(v51) = v71;
    _RDX = v66;
    v59 = **(_QWORD **)(v71 + 24);
  }
  else
  {
    v59 = *(_QWORD *)(v51 + 24);
  }
  if ( v59 > _RDX )
    goto LABEL_73;
LABEL_59:
  v15 = sub_33FAF80(*a1, 200, (unsigned int)&v90, v88, v89, v51, *(_OWORD *)*(_QWORD *)(v82 + 40));
  if ( v97 > 0x40 && v96 )
    j_j___libc_free_0_0(v96);
  if ( (unsigned int)v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
LABEL_14:
  if ( v90 )
    sub_B91220((__int64)&v90, v90);
  return v15;
}
