// Function: sub_1C85190
// Address: 0x1c85190
//
void __fastcall sub_1C85190(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r12
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rsi
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  _BYTE *v19; // rsi
  _BYTE *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  _BYTE *v26; // rsi
  _BYTE *v27; // rbx
  __int64 *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // r13
  __int64 *v32; // r12
  _QWORD *v33; // rax
  __int64 *v34; // rbx
  __int64 *v35; // r8
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // r13
  _QWORD *v39; // rax
  __int64 v40; // r15
  _QWORD *v41; // r13
  __int64 v42; // rsi
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  __int64 v45; // rbx
  __int64 v46; // rax
  unsigned int v47; // eax
  __int64 v48; // rsi
  __int64 v49; // rcx
  unsigned __int64 v50; // r8
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // esi
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  _QWORD *v58; // rax
  unsigned __int64 v59; // [rsp+8h] [rbp-138h]
  __int64 v60; // [rsp+10h] [rbp-130h]
  __int64 v61; // [rsp+18h] [rbp-128h]
  _QWORD *v62; // [rsp+30h] [rbp-110h]
  __int64 v63; // [rsp+38h] [rbp-108h]
  __int64 v64; // [rsp+40h] [rbp-100h]
  __int64 *v65; // [rsp+48h] [rbp-F8h]
  __int64 *v66; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v67; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v68; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v69; // [rsp+50h] [rbp-F0h]
  __int64 v71; // [rsp+60h] [rbp-E0h]
  __int64 v72; // [rsp+60h] [rbp-E0h]
  __int64 v73; // [rsp+60h] [rbp-E0h]
  __int64 v74; // [rsp+60h] [rbp-E0h]
  __int64 v75; // [rsp+60h] [rbp-E0h]
  int v76; // [rsp+6Ch] [rbp-D4h]
  _QWORD *v77; // [rsp+70h] [rbp-D0h]
  __int64 v78; // [rsp+70h] [rbp-D0h]
  __int64 *v79; // [rsp+78h] [rbp-C8h]
  unsigned int v80; // [rsp+8Ch] [rbp-B4h] BYREF
  __int64 *v81; // [rsp+90h] [rbp-B0h] BYREF
  _BYTE *v82; // [rsp+98h] [rbp-A8h]
  _BYTE *v83; // [rsp+A0h] [rbp-A0h]
  __int64 *v84; // [rsp+B0h] [rbp-90h] BYREF
  _BYTE *v85; // [rsp+B8h] [rbp-88h]
  _BYTE *v86; // [rsp+C0h] [rbp-80h]
  __int64 *v87; // [rsp+D0h] [rbp-70h] BYREF
  __int64 *v88; // [rsp+D8h] [rbp-68h]
  __int64 *v89; // [rsp+E0h] [rbp-60h]
  __int64 v90[2]; // [rsp+F0h] [rbp-50h] BYREF
  char v91; // [rsp+100h] [rbp-40h]
  char v92; // [rsp+101h] [rbp-3Fh]

  v63 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v62 = **(_QWORD ***)(a2 + 40);
  v2 = *(_QWORD *)(a2 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 24);
  v4 = v3 - 24;
  if ( !v3 )
    v4 = 0;
  v61 = v4;
  v76 = 101;
  if ( (unsigned __int8)sub_1C2F070(a2) )
  {
    v5 = a2;
    if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
    {
LABEL_6:
      v6 = *(_QWORD *)(a2 + 88);
      v7 = v6;
      goto LABEL_7;
    }
  }
  else
  {
    v76 = byte_4FBD8E0 == 0 ? 101 : 5;
    v5 = a2;
    if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
      goto LABEL_6;
  }
  v45 = v5;
  sub_15E08E0(v5, a2);
  v6 = *(_QWORD *)(v45 + 88);
  if ( (*(_BYTE *)(v45 + 18) & 1) != 0 )
    sub_15E08E0(v45, a2);
  v7 = *(_QWORD *)(v45 + 88);
LABEL_7:
  v64 = v7 + 40LL * *(_QWORD *)(a2 + 96);
  if ( v64 != v6 )
  {
LABEL_8:
    v8 = 1;
    v79 = *(__int64 **)v6;
    v9 = *(_QWORD *)v6;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v9 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v13 = *(_QWORD *)(v9 + 32);
          v9 = *(_QWORD *)(v9 + 24);
          v8 *= v13;
          continue;
        case 1:
          v10 = 16;
          break;
        case 2:
          v10 = 32;
          break;
        case 3:
        case 9:
          v10 = 64;
          break;
        case 4:
          v10 = 80;
          break;
        case 5:
        case 6:
          v10 = 128;
          break;
        case 7:
          v10 = 8 * (unsigned int)sub_15A9520(v63, 0);
          break;
        case 0xB:
          v10 = *(_DWORD *)(v9 + 8) >> 8;
          break;
        case 0xD:
          v10 = 8LL * *(_QWORD *)sub_15A9930(v63, v9);
          break;
        case 0xE:
          v14 = *(_QWORD *)(v9 + 32);
          v15 = 1;
          v16 = *(_QWORD *)(v9 + 24);
          v17 = (unsigned int)sub_15A9FE0(v63, v16);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v16 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v51 = *(_QWORD *)(v16 + 32);
                v16 = *(_QWORD *)(v16 + 24);
                v15 *= v51;
                continue;
              case 1:
                v46 = 16;
                break;
              case 2:
                v46 = 32;
                break;
              case 3:
              case 9:
                v46 = 64;
                break;
              case 4:
                v46 = 80;
                break;
              case 5:
              case 6:
                v46 = 128;
                break;
              case 7:
                v46 = 8 * (unsigned int)sub_15A9520(v63, 0);
                break;
              case 0xB:
                v46 = *(_DWORD *)(v16 + 8) >> 8;
                break;
              case 0xD:
                v46 = 8LL * *(_QWORD *)sub_15A9930(v63, v16);
                break;
              case 0xE:
                v78 = *(_QWORD *)(v16 + 32);
                v72 = *(_QWORD *)(v16 + 24);
                v47 = sub_15A9FE0(v63, v72);
                v48 = v72;
                v49 = 1;
                v50 = v47;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v48 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v55 = *(_QWORD *)(v48 + 32);
                      v48 = *(_QWORD *)(v48 + 24);
                      v49 *= v55;
                      continue;
                    case 1:
                      v52 = 16;
                      goto LABEL_92;
                    case 2:
                      v52 = 32;
                      goto LABEL_92;
                    case 3:
                    case 9:
                      v52 = 64;
                      goto LABEL_92;
                    case 4:
                      JUMPOUT(0x1C85A56);
                    case 5:
                    case 6:
                      JUMPOUT(0x1C85A4F);
                    case 7:
                      v67 = v50;
                      v53 = 0;
                      v73 = v49;
                      goto LABEL_95;
                    case 0xB:
                      v52 = *(_DWORD *)(v48 + 8) >> 8;
                      goto LABEL_92;
                    case 0xD:
                      v69 = v50;
                      v75 = v49;
                      v58 = (_QWORD *)sub_15A9930(v63, v48);
                      v49 = v75;
                      v50 = v69;
                      v52 = 8LL * *v58;
                      goto LABEL_92;
                    case 0xE:
                      v59 = v50;
                      v60 = v49;
                      v74 = *(_QWORD *)(v48 + 32);
                      v56 = *(_QWORD *)(v48 + 24);
                      v68 = (unsigned int)sub_15A9FE0(v63, v56);
                      v57 = sub_127FA20(v63, v56);
                      v49 = v60;
                      v50 = v59;
                      v52 = 8 * v74 * v68 * ((v68 + ((unsigned __int64)(v57 + 7) >> 3) - 1) / v68);
                      goto LABEL_92;
                    case 0xF:
                      v67 = v50;
                      v73 = v49;
                      v53 = *(_DWORD *)(v48 + 8) >> 8;
LABEL_95:
                      v54 = sub_15A9520(v63, v53);
                      v49 = v73;
                      v50 = v67;
                      v52 = (unsigned int)(8 * v54);
LABEL_92:
                      v46 = 8 * v78 * v50 * ((v50 + ((unsigned __int64)(v52 * v49 + 7) >> 3) - 1) / v50);
                      break;
                  }
                  break;
                }
                break;
              case 0xF:
                v46 = 8 * (unsigned int)sub_15A9520(v63, *(_DWORD *)(v16 + 8) >> 8);
                break;
            }
            break;
          }
          v10 = 8 * v17 * v14 * ((v17 + ((unsigned __int64)(v46 * v15 + 7) >> 3) - 1) / v17);
          break;
        case 0xF:
          v10 = 8 * (unsigned int)sub_15A9520(v63, *(_DWORD *)(v9 + 8) >> 8);
          break;
      }
      break;
    }
    if ( (unsigned int)dword_4FBD640 >= (unsigned __int64)(v10 * v8 + 7) >> 3 )
      goto LABEL_13;
    v11 = sub_3936750();
    v12 = sub_39371E0(a2, v11);
    sub_39367A0(v11);
    if ( v12 )
      goto LABEL_13;
    v81 = 0;
    v82 = 0;
    v83 = 0;
    v18 = sub_16471D0(v62, v76);
    v19 = v82;
    v90[0] = v18;
    if ( v82 == v83 )
    {
      sub_1278040((__int64)&v81, v82, v90);
      v20 = v82;
    }
    else
    {
      if ( v82 )
      {
        *(_QWORD *)v82 = v18;
        v19 = v82;
      }
      v20 = v19 + 8;
      v82 = v20;
    }
    v21 = sub_15E26F0(*(__int64 **)(a2 + 40), 3659, v81, (v20 - (_BYTE *)v81) >> 3);
    v84 = 0;
    v85 = 0;
    v22 = v21;
    v86 = 0;
    v23 = *(unsigned int *)(v6 + 32);
    v24 = sub_1643350(v62);
    v25 = sub_159C470(v24, v23, 0);
    v26 = v85;
    v90[0] = v25;
    if ( v85 == v86 )
    {
      sub_12879C0((__int64)&v84, v85, v90);
      v27 = v85;
    }
    else
    {
      if ( v85 )
      {
        *(_QWORD *)v85 = v25;
        v26 = v85;
      }
      v27 = v26 + 8;
      v85 = v26 + 8;
    }
    v28 = v84;
    v92 = 1;
    v91 = 3;
    v90[0] = (__int64)"ParamAddr";
    v29 = (v27 - (_BYTE *)v84) >> 3;
    v30 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
    v77 = sub_1648AB0(72, (int)v29 + 1, 0);
    if ( v77 )
    {
      sub_15F1EA0((__int64)v77, **(_QWORD **)(v30 + 16), 54, (__int64)&v77[-3 * v29 - 3], v29 + 1, v61);
      v77[7] = 0;
      sub_15F5B40((__int64)v77, v30, v22, v28, v29, (__int64)v90, 0, 0);
    }
    v80 = 0;
    if ( !(unsigned __int8)sub_1C2FF50(a2, *(_DWORD *)(v6 + 32) + 1, &v80) )
      v80 = sub_15A9FE0(v63, (__int64)v79);
    v87 = 0;
    v88 = 0;
    v89 = 0;
    v31 = *(_QWORD *)(v6 + 8);
    if ( !v31 )
      goto LABEL_66;
    v32 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v33 = sub_1648700(v31);
        if ( *((_BYTE *)v33 + 16) != 77 )
          break;
LABEL_43:
        v31 = *(_QWORD *)(v31 + 8);
        if ( !v31 )
          goto LABEL_47;
      }
      v90[0] = (__int64)v33;
      if ( v89 != v32 )
      {
        if ( v32 )
        {
          *v32 = (__int64)v33;
          v32 = v88;
        }
        v88 = ++v32;
        goto LABEL_43;
      }
      sub_17C2330((__int64)&v87, v32, v90);
      v31 = *(_QWORD *)(v31 + 8);
      v32 = v88;
      if ( !v31 )
      {
LABEL_47:
        v34 = v87;
        if ( v87 != v32 )
        {
          v71 = v6;
          v66 = v32;
          while ( 1 )
          {
            v36 = *v34;
            v37 = sub_1646BA0(v79, v76);
            v92 = 1;
            v38 = v37;
            v91 = 3;
            v90[0] = (__int64)"bitCast";
            v39 = sub_1648A60(56, 1u);
            v40 = (__int64)v39;
            if ( v39 )
              sub_15FD590((__int64)v39, (__int64)v77, v38, (__int64)v90, v36);
            v92 = 1;
            v90[0] = (__int64)"paramld";
            v91 = 3;
            v41 = sub_1648A60(64, 1u);
            if ( v41 )
              sub_15F90A0((__int64)v41, *(_QWORD *)(*(_QWORD *)v40 + 24LL), v40, (__int64)v90, 0, v80, v36);
            v42 = *(_QWORD *)(v36 + 48);
            v35 = v41 + 6;
            v90[0] = v42;
            if ( v42 )
              break;
            if ( v35 != v90 )
            {
              v43 = v41[6];
              if ( v43 )
                goto LABEL_60;
            }
LABEL_52:
            ++v34;
            sub_1648780(v36, v71, (__int64)v41);
            if ( v66 == v34 )
            {
              v6 = v71;
              v32 = v87;
              goto LABEL_64;
            }
          }
          sub_1623A60((__int64)v90, v42, 2);
          v35 = v41 + 6;
          if ( v41 + 6 == v90 )
          {
            if ( v90[0] )
              sub_161E7C0((__int64)v90, v90[0]);
            goto LABEL_52;
          }
          v43 = v41[6];
          if ( v43 )
          {
LABEL_60:
            v65 = v35;
            sub_161E7C0((__int64)v35, v43);
            v35 = v65;
          }
          v44 = (unsigned __int8 *)v90[0];
          v41[6] = v90[0];
          if ( v44 )
            sub_1623210((__int64)v90, v44, (__int64)v35);
          goto LABEL_52;
        }
LABEL_64:
        if ( v32 )
          j_j___libc_free_0(v32, (char *)v89 - (char *)v32);
LABEL_66:
        if ( v84 )
          j_j___libc_free_0(v84, v86 - (_BYTE *)v84);
        if ( v81 )
          j_j___libc_free_0(v81, v83 - (_BYTE *)v81);
LABEL_13:
        v6 += 40;
        if ( v6 == v64 )
          return;
        goto LABEL_8;
      }
    }
  }
}
