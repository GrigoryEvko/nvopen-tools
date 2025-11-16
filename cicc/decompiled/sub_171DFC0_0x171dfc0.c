// Function: sub_171DFC0
// Address: 0x171dfc0
//
__int64 __fastcall sub_171DFC0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r14
  __int64 v6; // r8
  bool v7; // al
  __int64 v8; // rcx
  __int64 v9; // r8
  char v10; // al
  __int64 *v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r12
  __int64 v15; // rcx
  __int64 v16; // r13
  char v17; // dl
  unsigned int v18; // eax
  _BYTE *v19; // rdi
  unsigned __int8 v20; // al
  __int64 v21; // rsi
  unsigned int v22; // ecx
  __int64 v23; // r9
  __int64 v24; // rbx
  __int64 *v25; // r10
  __int64 v26; // rax
  __int64 v27; // rdx
  _BYTE *v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rsi
  unsigned __int8 v31; // al
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned int v34; // eax
  __int64 *v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rsi
  int v40; // edi
  __int64 v41; // rdx
  _QWORD *v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rax
  char v47; // al
  __int64 v48; // [rsp+8h] [rbp-F8h]
  __int64 v49; // [rsp+8h] [rbp-F8h]
  __int64 *v51; // [rsp+10h] [rbp-F0h]
  __int64 *v52; // [rsp+10h] [rbp-F0h]
  __int64 v53; // [rsp+18h] [rbp-E8h]
  __int64 v54; // [rsp+18h] [rbp-E8h]
  __int64 v55; // [rsp+18h] [rbp-E8h]
  __int64 v56; // [rsp+18h] [rbp-E8h]
  char v57; // [rsp+2Eh] [rbp-D2h] BYREF
  char v58; // [rsp+2Fh] [rbp-D1h] BYREF
  __int64 *v59; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v61; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v62; // [rsp+48h] [rbp-B8h] BYREF
  const void *v63; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v64; // [rsp+58h] [rbp-A8h]
  const void *v65; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v66; // [rsp+68h] [rbp-98h]
  __int64 v67; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v68; // [rsp+78h] [rbp-88h]
  const void *v69; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v70; // [rsp+88h] [rbp-78h]
  __int64 v71[2]; // [rsp+90h] [rbp-70h] BYREF
  char v72; // [rsp+A0h] [rbp-60h]
  char v73; // [rsp+A1h] [rbp-5Fh]
  __int64 **v74; // [rsp+B0h] [rbp-50h] BYREF
  __int64 *v75; // [rsp+B8h] [rbp-48h] BYREF
  __int16 v76; // [rsp+C0h] [rbp-40h]

  v5 = *(_QWORD *)(a2 - 48);
  v6 = *(_QWORD *)(a2 - 24);
  v64 = 1;
  v53 = v6;
  v63 = 0;
  v66 = 1;
  v65 = 0;
  v7 = sub_1719E80(v5, &v59, (__int64)&v63, &v57);
  v9 = v53;
  if ( !v7 || (v10 = sub_1719B90(v53, &v60, (__int64)&v65, v8), v9 = v53, !v10) )
  {
    if ( !sub_1719E80(v9, &v59, (__int64)&v63, &v57) || !(unsigned __int8)sub_1719B90(v5, &v60, (__int64)&v65, v15) )
      goto LABEL_8;
    if ( v64 > 0x40 )
      goto LABEL_4;
LABEL_19:
    if ( v63 == v65 )
      goto LABEL_5;
    goto LABEL_8;
  }
  if ( v64 <= 0x40 )
    goto LABEL_19;
LABEL_4:
  if ( sub_16A5220((__int64)&v63, &v65) )
  {
LABEL_5:
    v68 = 1;
    v11 = &v61;
    v67 = 0;
    if ( !sub_1719E80(v60, &v61, (__int64)&v67, &v58) || v57 != v58 )
    {
LABEL_7:
      if ( v68 > 0x40 && v67 )
        j_j___libc_free_0_0(v67);
      goto LABEL_8;
    }
    v16 = v61;
    v70 = 1;
    v69 = 0;
    v17 = *(_BYTE *)(v61 + 16);
    if ( !v57 )
    {
      if ( v17 == 41 )
      {
        if ( !*(_QWORD *)(v61 - 48) )
          goto LABEL_29;
        v62 = *(__int64 **)(v61 - 48);
        v19 = *(_BYTE **)(v61 - 24);
        v20 = v19[16];
        if ( v20 != 13 )
        {
          v36 = *(_QWORD *)v19;
          if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) != 16 || v20 > 0x10u )
            goto LABEL_29;
LABEL_87:
          v37 = sub_15A1020(v19, (__int64)&v61, v36, v12);
          v11 = (__int64 *)v37;
          if ( v37 && *(_BYTE *)(v37 + 16) == 13 )
          {
            v21 = v37 + 24;
            v71[0] = v37 + 24;
            if ( v70 > 0x40 )
              goto LABEL_37;
            goto LABEL_36;
          }
          v47 = *(_BYTE *)(v16 + 16);
          v74 = &v62;
          v75 = v71;
          if ( v47 != 48 )
          {
            if ( v47 != 5 )
              goto LABEL_29;
            goto LABEL_28;
          }
LABEL_71:
          if ( !*(_QWORD *)(v16 - 48) )
            goto LABEL_29;
          v62 = *(__int64 **)(v16 - 48);
          if ( !(unsigned __int8)sub_13D2630(&v75, *(_BYTE **)(v16 - 24)) )
            goto LABEL_29;
          v33 = v71[0];
LABEL_74:
          v34 = *(_DWORD *)(v33 + 8);
          LODWORD(v75) = v34;
          if ( v34 > 0x40 )
            sub_16A4EF0((__int64)&v74, 1, 0);
          else
            v74 = (__int64 **)((0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & 1);
          if ( v70 > 0x40 && v69 )
            j_j___libc_free_0_0(v69);
          v69 = v74;
          v70 = (unsigned int)v75;
          sub_16A7E20((__int64)&v69, v71[0]);
LABEL_38:
          if ( v59 != v62 )
            goto LABEL_29;
          goto LABEL_39;
        }
      }
      else
      {
        if ( v17 != 5 )
        {
          v74 = &v62;
          v75 = v71;
          if ( v17 != 48 )
            goto LABEL_29;
          goto LABEL_71;
        }
        if ( *(_WORD *)(v61 + 18) != 17 )
          goto LABEL_27;
        v38 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
        v12 = 4 * v38;
        if ( !*(_QWORD *)(v61 - 24 * v38) )
          goto LABEL_27;
        v62 = *(__int64 **)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
        v36 = 1 - v38;
        v19 = *(_BYTE **)(v61 + 24 * (1 - v38));
        if ( v19[16] != 13 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) == 16 )
            goto LABEL_87;
LABEL_27:
          v74 = &v62;
          v75 = v71;
LABEL_28:
          if ( *(_WORD *)(v16 + 18) != 24 )
            goto LABEL_29;
          v43 = *(_DWORD *)(v16 + 20) & 0xFFFFFFF;
          if ( !*(_QWORD *)(v16 - 24 * v43) )
            goto LABEL_29;
          v62 = *(__int64 **)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
          v44 = *(_BYTE **)(v16 + 24 * (1 - v43));
          if ( v44[16] == 13 )
          {
            v33 = (__int64)(v44 + 24);
            v71[0] = (__int64)(v44 + 24);
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v44 + 8LL) != 16 )
              goto LABEL_29;
            v46 = sub_15A1020(v44, (__int64)v11, 1 - v43, 4 * v43);
            if ( !v46 || *(_BYTE *)(v46 + 16) != 13 )
              goto LABEL_29;
            *v75 = v46 + 24;
            v33 = v71[0];
          }
          goto LABEL_74;
        }
      }
      v21 = (__int64)(v19 + 24);
      v71[0] = (__int64)(v19 + 24);
LABEL_36:
      v22 = *(_DWORD *)(v21 + 8);
      if ( v22 > 0x40 )
      {
LABEL_37:
        sub_16A51C0((__int64)&v69, v21);
        goto LABEL_38;
      }
      v45 = *(_QWORD *)v21;
      v70 = *(_DWORD *)(v21 + 8);
      v69 = (const void *)(v45 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22));
LABEL_57:
      if ( v62 != v59 )
        goto LABEL_7;
LABEL_39:
      if ( v64 <= 0x40 )
      {
        if ( v63 == v69 )
        {
LABEL_41:
          if ( v57 )
            sub_16AA420((__int64)&v74, (__int64)&v63, (__int64)&v67, (bool *)v71);
          else
            sub_16AA580((__int64)&v74, (__int64)&v63, (__int64)&v67, (bool *)v71);
          sub_135E100((__int64 *)&v74);
          if ( !LOBYTE(v71[0]) )
          {
            sub_16A7B50((__int64)&v74, (__int64)&v63, &v67);
            v54 = sub_159C0E0(*(__int64 **)*v59, (__int64)&v74);
            sub_135E100((__int64 *)&v74);
            v73 = 1;
            v23 = v54;
            v24 = *(_QWORD *)(a1 + 8);
            if ( v57 )
            {
              v25 = v59;
              v72 = 3;
              v71[0] = (__int64)"srem";
              if ( *((_BYTE *)v59 + 16) <= 0x10u && *(_BYTE *)(v54 + 16) <= 0x10u )
              {
                v48 = v54;
                v51 = v59;
                v55 = sub_15A2A30((__int64 *)0x15, v59, v54, 0, 0, a3, a4, a5);
                v13 = sub_14DBA30(v55, *(_QWORD *)(v24 + 96), 0);
                if ( v13 )
                  goto LABEL_48;
                v25 = v51;
                v23 = v48;
                v13 = v55;
                if ( v55 )
                  goto LABEL_48;
              }
              v39 = v25;
              v76 = 257;
              v40 = 21;
              v41 = v23;
              goto LABEL_97;
            }
            v35 = v59;
            v72 = 3;
            v71[0] = (__int64)"urem";
            if ( *((_BYTE *)v59 + 16) <= 0x10u && *(_BYTE *)(v54 + 16) <= 0x10u )
            {
              v49 = v54;
              v52 = v59;
              v56 = sub_15A2A30((__int64 *)0x14, v59, v54, 0, 0, a3, a4, a5);
              v13 = sub_14DBA30(v56, *(_QWORD *)(v24 + 96), 0);
              if ( v13 )
              {
LABEL_48:
                sub_135E100((__int64 *)&v69);
                sub_135E100(&v67);
                goto LABEL_9;
              }
              v35 = v52;
              v23 = v49;
              if ( v56 )
              {
                v13 = v56;
                goto LABEL_48;
              }
            }
            v41 = v23;
            v76 = 257;
            v39 = v35;
            v40 = 20;
LABEL_97:
            v42 = (_QWORD *)sub_15FB440(v40, v39, v41, (__int64)&v74, 0);
            v13 = (__int64)sub_171D920(v24, v42, v71);
            goto LABEL_48;
          }
        }
      }
      else if ( sub_16A5220((__int64)&v63, &v69) )
      {
        goto LABEL_41;
      }
LABEL_29:
      v18 = v70;
      goto LABEL_30;
    }
    if ( v17 == 42 )
    {
      if ( !*(_QWORD *)(v61 - 48) )
        goto LABEL_7;
      v62 = *(__int64 **)(v61 - 48);
      v28 = *(_BYTE **)(v61 - 24);
      v31 = v28[16];
      if ( v31 == 13 )
      {
LABEL_54:
        v21 = (__int64)(v28 + 24);
        v71[0] = (__int64)(v28 + 24);
LABEL_55:
        v29 = *(_DWORD *)(v21 + 8);
        if ( v29 > 0x40 )
          goto LABEL_37;
        v30 = *(_QWORD *)v21;
        v70 = v29;
        v69 = (const void *)(v30 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29));
        goto LABEL_57;
      }
      v27 = *(_QWORD *)v28;
      if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) != 16 || v31 > 0x10u )
        goto LABEL_7;
    }
    else
    {
      if ( v17 != 5 )
        goto LABEL_7;
      if ( *(_WORD *)(v61 + 18) != 18 )
        goto LABEL_7;
      v26 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      v12 = 4 * v26;
      if ( !*(_QWORD *)(v61 - 24 * v26) )
        goto LABEL_7;
      v62 = *(__int64 **)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
      v27 = 1 - v26;
      v28 = *(_BYTE **)(v61 + 24 * (1 - v26));
      if ( v28[16] == 13 )
        goto LABEL_54;
      if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) != 16 )
        goto LABEL_7;
    }
    v32 = sub_15A1020(v28, (__int64)&v61, v27, v12);
    v18 = v70;
    if ( !v32 || *(_BYTE *)(v32 + 16) != 13 )
    {
LABEL_30:
      if ( v18 > 0x40 && v69 )
        j_j___libc_free_0_0(v69);
      goto LABEL_7;
    }
    v21 = v32 + 24;
    v71[0] = v21;
    if ( v70 > 0x40 )
      goto LABEL_37;
    goto LABEL_55;
  }
LABEL_8:
  v13 = 0;
LABEL_9:
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  if ( v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  return v13;
}
