// Function: sub_384FD60
// Address: 0x384fd60
//
__int64 __fastcall sub_384FD60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 *v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned int v10; // r8d
  unsigned __int64 v11; // r12
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r14
  int v15; // eax
  __int64 v16; // r11
  _QWORD *v17; // rbx
  __int64 v18; // rsi
  char v19; // al
  int v21; // edx
  __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r9
  __int64 v27; // r9
  unsigned int v28; // eax
  __int64 v29; // r9
  __int64 v30; // r11
  __int64 v31; // rcx
  unsigned __int64 v32; // r8
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // eax
  __int64 v38; // rdi
  _QWORD *v39; // rax
  __int64 v40; // rdi
  unsigned int v41; // eax
  __int64 v42; // rsi
  __int64 v43; // r9
  __int64 v44; // r11
  unsigned __int64 v45; // r10
  __int64 v46; // rax
  int v47; // eax
  int v48; // r8d
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // esi
  int v52; // eax
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // [rsp+8h] [rbp-D8h]
  __int64 v58; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v59; // [rsp+18h] [rbp-C8h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+28h] [rbp-B8h]
  __int64 v63; // [rsp+30h] [rbp-B0h]
  __int64 v64; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v65; // [rsp+38h] [rbp-A8h]
  unsigned int v67; // [rsp+50h] [rbp-90h]
  __int64 v68; // [rsp+50h] [rbp-90h]
  unsigned __int64 v69; // [rsp+50h] [rbp-90h]
  unsigned __int64 v70; // [rsp+50h] [rbp-90h]
  unsigned __int64 v71; // [rsp+50h] [rbp-90h]
  __int64 v72; // [rsp+58h] [rbp-88h]
  __int64 v73; // [rsp+58h] [rbp-88h]
  unsigned __int64 v74; // [rsp+58h] [rbp-88h]
  unsigned __int64 v75; // [rsp+58h] [rbp-88h]
  __int64 v76; // [rsp+58h] [rbp-88h]
  __int64 v77; // [rsp+58h] [rbp-88h]
  __int64 v78; // [rsp+58h] [rbp-88h]
  __int64 v79; // [rsp+58h] [rbp-88h]
  __int64 v80; // [rsp+60h] [rbp-80h]
  __int64 v81; // [rsp+60h] [rbp-80h]
  __int64 v82; // [rsp+60h] [rbp-80h]
  __int64 v83; // [rsp+60h] [rbp-80h]
  __int64 v84; // [rsp+60h] [rbp-80h]
  unsigned int v86; // [rsp+74h] [rbp-6Ch]
  __int64 v87; // [rsp+78h] [rbp-68h]
  unsigned __int64 v88; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v89; // [rsp+88h] [rbp-58h]
  unsigned __int64 v90; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v91; // [rsp+98h] [rbp-48h]
  unsigned __int64 v92; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v93; // [rsp+A8h] [rbp-38h]

  v87 = a2;
  v86 = sub_15A95F0(*(_QWORD *)(a1 + 40), *(_QWORD *)a2);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v4 = (__int64 *)(v3 + 24);
  v5 = sub_16348C0(a2) | 4;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v87 = *(_QWORD *)(a2 - 8) + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( v4 == (__int64 *)v87 )
    return 1;
  v6 = v4;
  v65 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v86;
  while ( 1 )
  {
    v7 = *v6;
    v8 = *v6;
    if ( *(_BYTE *)(*v6 + 16) == 13 )
      goto LABEL_8;
    v21 = *(_DWORD *)(a1 + 160);
    if ( !v21 )
      return 0;
    v22 = *(_QWORD *)(a1 + 144);
    v23 = v21 - 1;
    v24 = v23 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v8 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v7 != *v25 )
      break;
LABEL_32:
    v8 = v25[1];
    if ( !v8 || *(_BYTE *)(v8 + 16) != 13 )
      return 0;
LABEL_8:
    v9 = v5;
    v10 = *(_DWORD *)(v8 + 32);
    v11 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = (__int64 *)(v8 + 24);
    v13 = v11;
    v14 = (v9 >> 2) & 1;
    if ( v10 <= 0x40 )
    {
      if ( !*(_QWORD *)(v8 + 24) )
        goto LABEL_24;
    }
    else
    {
      v67 = *(_DWORD *)(v8 + 32);
      v72 = v8;
      v15 = sub_16A57B0(v8 + 24);
      v10 = v67;
      v8 = v72;
      if ( v67 == v15 )
        goto LABEL_24;
    }
    v16 = *(_QWORD *)(a1 + 40);
    if ( (_BYTE)v14 )
    {
      v27 = v11;
      if ( v11 )
        goto LABEL_36;
    }
    else if ( v11 )
    {
      v17 = *(_QWORD **)(v8 + 24);
      if ( v10 > 0x40 )
        v17 = (_QWORD *)*v17;
      v18 = *(_QWORD *)(sub_15A9930(*(_QWORD *)(a1 + 40), v11) + 8LL * (unsigned int)v17 + 16);
      v93 = v86;
      if ( v86 > 0x40 )
        sub_16A4EF0((__int64)&v92, v18, 0);
      else
        v92 = v65 & v18;
      sub_16A7200(a3, (__int64 *)&v92);
      if ( v93 > 0x40 && v92 )
        j_j___libc_free_0_0(v92);
LABEL_19:
      v13 = sub_1643D30(v11, *v6);
      v19 = *(_BYTE *)(v13 + 8);
      if ( ((v19 - 14) & 0xFD) != 0 )
        goto LABEL_27;
      goto LABEL_20;
    }
    v81 = *(_QWORD *)(a1 + 40);
    v33 = sub_1643D30(0, v7);
    v16 = v81;
    v27 = v33;
LABEL_36:
    v73 = v27;
    v80 = v16;
    v28 = sub_15A9FE0(v16, v27);
    v29 = v73;
    v30 = v80;
    v31 = 1;
    v32 = v28;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v29 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v46 = *(_QWORD *)(v29 + 32);
          v29 = *(_QWORD *)(v29 + 24);
          v31 *= v46;
          continue;
        case 1:
          v34 = 16;
          goto LABEL_42;
        case 2:
          v34 = 32;
          goto LABEL_42;
        case 3:
        case 9:
          v34 = 64;
          goto LABEL_42;
        case 4:
          v34 = 80;
          goto LABEL_42;
        case 5:
        case 6:
          v34 = 128;
          goto LABEL_42;
        case 7:
          v74 = v32;
          v36 = 0;
          v82 = v31;
          goto LABEL_56;
        case 0xB:
          v34 = *(_DWORD *)(v29 + 8) >> 8;
          goto LABEL_42;
        case 0xD:
          v38 = v80;
          v75 = v32;
          v83 = v31;
          v39 = (_QWORD *)sub_15A9930(v38, v29);
          v31 = v83;
          v32 = v75;
          v34 = 8LL * *v39;
          goto LABEL_42;
        case 0xE:
          v40 = v80;
          v61 = v32;
          v63 = v31;
          v68 = *(_QWORD *)(v29 + 24);
          v76 = v80;
          v84 = *(_QWORD *)(v29 + 32);
          v41 = sub_15A9FE0(v40, v68);
          v32 = v61;
          v42 = v68;
          v43 = 1;
          v31 = v63;
          v44 = v76;
          v45 = v41;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v42 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v50 = *(_QWORD *)(v42 + 32);
                v42 = *(_QWORD *)(v42 + 24);
                v43 *= v50;
                continue;
              case 1:
                v49 = 16;
                goto LABEL_71;
              case 2:
                v49 = 32;
                goto LABEL_71;
              case 3:
              case 9:
                v49 = 64;
                goto LABEL_71;
              case 4:
                v49 = 80;
                goto LABEL_71;
              case 5:
              case 6:
                v49 = 128;
                goto LABEL_71;
              case 7:
                v51 = 0;
                v69 = v45;
                v77 = v43;
                goto LABEL_78;
              case 0xB:
                v49 = *(_DWORD *)(v42 + 8) >> 8;
                goto LABEL_71;
              case 0xD:
                v55 = v76;
                v71 = v45;
                v79 = v43;
                v56 = (_QWORD *)sub_15A9930(v55, v42);
                v43 = v79;
                v45 = v71;
                v31 = v63;
                v32 = v61;
                v49 = 8LL * *v56;
                goto LABEL_71;
              case 0xE:
                v53 = v76;
                v57 = v61;
                v58 = v63;
                v59 = v45;
                v60 = v43;
                v62 = *(_QWORD *)(v42 + 24);
                v64 = v76;
                v78 = *(_QWORD *)(v42 + 32);
                v70 = (unsigned int)sub_15A9FE0(v53, v62);
                v54 = sub_127FA20(v64, v62);
                v43 = v60;
                v45 = v59;
                v31 = v58;
                v32 = v57;
                v49 = 8 * v70 * v78 * ((v70 + ((unsigned __int64)(v54 + 7) >> 3) - 1) / v70);
                goto LABEL_71;
              case 0xF:
                v69 = v45;
                v51 = *(_DWORD *)(v42 + 8) >> 8;
                v77 = v43;
LABEL_78:
                v52 = sub_15A9520(v44, v51);
                v43 = v77;
                v45 = v69;
                v31 = v63;
                v32 = v61;
                v49 = (unsigned int)(8 * v52);
LABEL_71:
                v34 = 8 * v84 * v45 * ((v45 + ((unsigned __int64)(v49 * v43 + 7) >> 3) - 1) / v45);
                break;
            }
            goto LABEL_42;
          }
        case 0xF:
          v74 = v32;
          v82 = v31;
          v36 = *(_DWORD *)(v29 + 8) >> 8;
LABEL_56:
          v37 = sub_15A9520(v30, v36);
          v31 = v82;
          v32 = v74;
          v34 = (unsigned int)(8 * v37);
LABEL_42:
          v89 = v86;
          v35 = v32 * ((v32 + ((unsigned __int64)(v34 * v31 + 7) >> 3) - 1) / v32);
          if ( v86 > 0x40 )
            sub_16A4EF0((__int64)&v88, v35, 0);
          else
            v88 = v65 & v35;
          sub_16A5D70((__int64)&v90, v12, v86);
          sub_16A7B50((__int64)&v92, (__int64)&v90, (__int64 *)&v88);
          sub_16A7200(a3, (__int64 *)&v92);
          if ( v93 > 0x40 && v92 )
            j_j___libc_free_0_0(v92);
          if ( v91 > 0x40 && v90 )
            j_j___libc_free_0_0(v90);
          if ( v89 > 0x40 && v88 )
            j_j___libc_free_0_0(v88);
          break;
      }
      break;
    }
LABEL_24:
    if ( !(_BYTE)v14 || !v11 )
      goto LABEL_19;
    v19 = *(_BYTE *)(v11 + 8);
    if ( ((v19 - 14) & 0xFD) != 0 )
    {
LABEL_27:
      v5 = 0;
      if ( v19 == 13 )
        v5 = v13;
      goto LABEL_21;
    }
LABEL_20:
    v5 = *(_QWORD *)(v13 + 24) | 4LL;
LABEL_21:
    v6 += 3;
    if ( (__int64 *)v87 == v6 )
      return 1;
  }
  v47 = 1;
  while ( v26 != -8 )
  {
    v48 = v47 + 1;
    v24 = v23 & (v47 + v24);
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v7 == *v25 )
      goto LABEL_32;
    v47 = v48;
  }
  return 0;
}
