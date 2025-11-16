// Function: sub_1634900
// Address: 0x1634900
//
__int64 __fastcall sub_1634900(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // r14
  int v15; // eax
  _QWORD *v16; // r13
  __int64 v17; // rax
  unsigned int v18; // ecx
  __int64 v19; // rsi
  char v20; // al
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rcx
  unsigned __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned int v28; // ecx
  unsigned __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned __int64 v33; // r9
  unsigned int v34; // esi
  int v35; // eax
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rax
  _QWORD *v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-B0h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  unsigned int v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+18h] [rbp-98h]
  unsigned __int64 v55; // [rsp+18h] [rbp-98h]
  unsigned __int64 v56; // [rsp+18h] [rbp-98h]
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+20h] [rbp-90h]
  unsigned __int64 v59; // [rsp+20h] [rbp-90h]
  unsigned __int64 v60; // [rsp+20h] [rbp-90h]
  __int64 v61; // [rsp+20h] [rbp-90h]
  __int64 v62; // [rsp+20h] [rbp-90h]
  __int64 v63; // [rsp+20h] [rbp-90h]
  __int64 v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+28h] [rbp-88h]
  __int64 v66; // [rsp+28h] [rbp-88h]
  __int64 v67; // [rsp+28h] [rbp-88h]
  __int64 v69; // [rsp+48h] [rbp-68h]
  __int64 v70; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v71; // [rsp+58h] [rbp-58h]
  unsigned __int64 v72; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v73; // [rsp+68h] [rbp-48h]
  unsigned __int64 v74; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v75; // [rsp+78h] [rbp-38h]

  v69 = a1;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(a1 - 8);
  else
    v4 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v5 = (_QWORD *)(v4 + 24);
  v7 = sub_16348C0(a1) | 4;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v69 = *(_QWORD *)(a1 - 8) + 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( v5 == (_QWORD *)v69 )
    return 1;
  while ( 1 )
  {
    v8 = *v5;
    if ( *(_BYTE *)(*v5 + 16LL) != 13 )
      return 0;
    v9 = v7;
    v10 = *(unsigned int *)(v8 + 32);
    v11 = v7 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = v8 + 24;
    v13 = v11;
    v14 = (v9 >> 2) & 1;
    if ( (unsigned int)v10 <= 0x40 )
    {
      if ( !*(_QWORD *)(v8 + 24) )
        goto LABEL_23;
    }
    else
    {
      v52 = *(_DWORD *)(v8 + 32);
      v57 = *v5;
      v64 = v8 + 24;
      v15 = sub_16A57B0(v8 + 24);
      v10 = v52;
      v12 = v64;
      v8 = v57;
      if ( v52 == v15 )
        goto LABEL_23;
    }
    if ( (_BYTE)v14 )
    {
      sub_16A5D70(&v70, v12, *(unsigned int *)(a3 + 8), v10, v6);
      v22 = v11;
      if ( v11 )
        goto LABEL_32;
    }
    else
    {
      if ( v11 )
      {
        v16 = *(_QWORD **)(v8 + 24);
        if ( (unsigned int)v10 > 0x40 )
          v16 = (_QWORD *)*v16;
        v17 = sub_15A9930(a2, v11);
        v18 = *(_DWORD *)(a3 + 8);
        v19 = *(_QWORD *)(v17 + 8LL * (unsigned int)v16 + 16);
        v75 = v18;
        if ( v18 <= 0x40 )
          v74 = v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18);
        else
          sub_16A4EF0(&v74, v19, 0);
        sub_16A7200(a3, &v74);
        if ( v75 > 0x40 && v74 )
          j_j___libc_free_0_0(v74);
LABEL_18:
        v13 = sub_1643D30(v11, *v5);
        v20 = *(_BYTE *)(v13 + 8);
        if ( ((v20 - 14) & 0xFD) != 0 )
          goto LABEL_26;
        goto LABEL_19;
      }
      sub_16A5D70(&v70, v12, *(unsigned int *)(a3 + 8), v10, v6);
    }
    v22 = sub_1643D30(0, *v5);
LABEL_32:
    v23 = sub_15A9FE0(a2, v22);
    v24 = 1;
    v25 = v23;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v22 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v36 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v24 *= v36;
          continue;
        case 1:
          v26 = 16;
          goto LABEL_38;
        case 2:
          v26 = 32;
          goto LABEL_38;
        case 3:
        case 9:
          v26 = 64;
          goto LABEL_38;
        case 4:
          v26 = 80;
          goto LABEL_38;
        case 5:
        case 6:
          v26 = 128;
          goto LABEL_38;
        case 7:
          v59 = v25;
          v34 = 0;
          v66 = v24;
          goto LABEL_56;
        case 0xB:
          v26 = *(_DWORD *)(v22 + 8) >> 8;
          goto LABEL_38;
        case 0xD:
          v60 = v25;
          v67 = v24;
          v37 = (_QWORD *)sub_15A9930(a2, v22);
          v24 = v67;
          v25 = v60;
          v26 = 8LL * *v37;
          goto LABEL_38;
        case 0xE:
          v48 = v25;
          v53 = v24;
          v65 = *(_QWORD *)(v22 + 32);
          v58 = *(_QWORD *)(v22 + 24);
          v30 = sub_15A9FE0(a2, v58);
          v25 = v48;
          v31 = 1;
          v32 = v58;
          v24 = v53;
          v33 = v30;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v32 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v43 = *(_QWORD *)(v32 + 32);
                v32 = *(_QWORD *)(v32 + 24);
                v31 *= v43;
                continue;
              case 1:
                v38 = 16;
                goto LABEL_63;
              case 2:
                v38 = 32;
                goto LABEL_63;
              case 3:
              case 9:
                v38 = 64;
                goto LABEL_63;
              case 4:
                v38 = 80;
                goto LABEL_63;
              case 5:
              case 6:
                v38 = 128;
                goto LABEL_63;
              case 7:
                v47 = v48;
                v41 = 0;
                v51 = v53;
                v56 = v33;
                v63 = v31;
                goto LABEL_70;
              case 0xB:
                v38 = *(_DWORD *)(v32 + 8) >> 8;
                goto LABEL_63;
              case 0xD:
                v46 = v48;
                v50 = v53;
                v55 = v33;
                v62 = v31;
                v40 = (_QWORD *)sub_15A9930(a2, v32);
                v31 = v62;
                v33 = v55;
                v24 = v50;
                v25 = v46;
                v38 = 8LL * *v40;
                goto LABEL_63;
              case 0xE:
                v44 = v48;
                v45 = v53;
                v49 = v33;
                v54 = v31;
                v61 = *(_QWORD *)(v32 + 32);
                v39 = sub_12BE0A0(a2, *(_QWORD *)(v32 + 24));
                v31 = v54;
                v33 = v49;
                v24 = v45;
                v25 = v44;
                v38 = 8 * v61 * v39;
                goto LABEL_63;
              case 0xF:
                v47 = v48;
                v51 = v53;
                v56 = v33;
                v41 = *(_DWORD *)(v32 + 8) >> 8;
                v63 = v31;
LABEL_70:
                v42 = sub_15A9520(a2, v41);
                v31 = v63;
                v33 = v56;
                v24 = v51;
                v25 = v47;
                v38 = (unsigned int)(8 * v42);
LABEL_63:
                v26 = 8 * v65 * v33 * ((v33 + ((unsigned __int64)(v38 * v31 + 7) >> 3) - 1) / v33);
                break;
            }
            goto LABEL_38;
          }
        case 0xF:
          v59 = v25;
          v66 = v24;
          v34 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_56:
          v35 = sub_15A9520(a2, v34);
          v24 = v66;
          v25 = v59;
          v26 = (unsigned int)(8 * v35);
LABEL_38:
          v27 = v25 + ((unsigned __int64)(v26 * v24 + 7) >> 3) - 1;
          v28 = *(_DWORD *)(a3 + 8);
          v73 = v28;
          v29 = v25 * (v27 / v25);
          if ( v28 > 0x40 )
            sub_16A4EF0(&v72, v29, 0);
          else
            v72 = v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28);
          sub_16A7B50(&v74, &v70, &v72);
          sub_16A7200(a3, &v74);
          if ( v75 > 0x40 && v74 )
            j_j___libc_free_0_0(v74);
          if ( v73 > 0x40 && v72 )
            j_j___libc_free_0_0(v72);
          if ( v71 > 0x40 && v70 )
            j_j___libc_free_0_0(v70);
          break;
      }
      break;
    }
LABEL_23:
    if ( !(_BYTE)v14 || !v11 )
      goto LABEL_18;
    v20 = *(_BYTE *)(v11 + 8);
    if ( ((v20 - 14) & 0xFD) != 0 )
    {
LABEL_26:
      v7 = 0;
      if ( v20 == 13 )
        v7 = v13;
      goto LABEL_20;
    }
LABEL_19:
    v7 = *(_QWORD *)(v13 + 24) | 4LL;
LABEL_20:
    v5 += 3;
    if ( (_QWORD *)v69 == v5 )
      return 1;
  }
}
