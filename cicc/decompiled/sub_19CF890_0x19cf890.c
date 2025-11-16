// Function: sub_19CF890
// Address: 0x19cf890
//
__int64 __fastcall sub_19CF890(__int64 a1, int a2, _BYTE *a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 *v6; // r13
  __int64 v7; // rdx
  __int64 *v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rax
  char v12; // dl
  bool v13; // zf
  int v14; // r15d
  __int64 v15; // r12
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // r14
  unsigned int v19; // ebx
  int v20; // eax
  bool v21; // al
  __int64 v22; // rbx
  unsigned __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rcx
  unsigned __int64 v30; // r10
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 *v34; // rdx
  unsigned __int64 v35; // rax
  __int64 v36; // r10
  char v37; // al
  __int64 v38; // rax
  _QWORD *v39; // rdx
  __int64 v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  unsigned int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // r8
  unsigned __int64 v46; // r11
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-A0h]
  __int64 v53; // [rsp+8h] [rbp-98h]
  unsigned __int64 v54; // [rsp+10h] [rbp-90h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  __int64 v57; // [rsp+20h] [rbp-80h]
  __int64 v58; // [rsp+20h] [rbp-80h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+28h] [rbp-78h]
  __int64 v61; // [rsp+28h] [rbp-78h]
  unsigned __int64 v62; // [rsp+28h] [rbp-78h]
  unsigned __int64 v63; // [rsp+30h] [rbp-70h]
  __int64 v64; // [rsp+30h] [rbp-70h]
  unsigned __int64 v65; // [rsp+30h] [rbp-70h]
  unsigned __int64 v66; // [rsp+30h] [rbp-70h]
  __int64 v67; // [rsp+30h] [rbp-70h]
  __int64 v68; // [rsp+38h] [rbp-68h]
  __int64 v69; // [rsp+38h] [rbp-68h]
  __int64 v70; // [rsp+38h] [rbp-68h]
  __int64 v71; // [rsp+38h] [rbp-68h]
  __int64 v72; // [rsp+38h] [rbp-68h]
  __int64 v73; // [rsp+38h] [rbp-68h]
  __int64 v74; // [rsp+40h] [rbp-60h]
  __int64 v75; // [rsp+40h] [rbp-60h]
  __int64 v76; // [rsp+40h] [rbp-60h]
  __int64 v77; // [rsp+40h] [rbp-60h]
  __int64 v78; // [rsp+40h] [rbp-60h]
  __int64 v79; // [rsp+40h] [rbp-60h]
  __int64 v80; // [rsp+40h] [rbp-60h]
  int v83; // [rsp+5Ch] [rbp-44h]
  __int64 *v84; // [rsp+60h] [rbp-40h]
  __int64 v85; // [rsp+68h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v6 = (__int64 *)(v5 + 24);
  v7 = sub_16348C0(a1) | 4;
  if ( a2 != 1 )
  {
    v8 = (__int64 *)(v5 + 24);
    v9 = v5 + 24LL * (unsigned int)(a2 - 2) + 48;
    do
    {
      while ( 1 )
      {
        v10 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        v11 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v7 & 4) == 0 || !v10 )
          v11 = sub_1643D30(v10, *v8);
        v12 = *(_BYTE *)(v11 + 8);
        if ( ((v12 - 14) & 0xFD) != 0 )
          break;
        v8 += 3;
        v7 = *(_QWORD *)(v11 + 24) | 4LL;
        if ( v8 == (__int64 *)v9 )
          goto LABEL_12;
      }
      v13 = v12 == 13;
      v7 = 0;
      if ( v13 )
        v7 = v11;
      v8 += 3;
    }
    while ( v8 != (__int64 *)v9 );
LABEL_12:
    v6 += 3 * (unsigned int)(a2 - 1);
  }
  v83 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( a2 == v83 )
    return 0;
  v84 = v6;
  v14 = a2 + 1;
  v15 = v7;
  v16 = a1;
  v85 = 0;
  while ( 1 )
  {
    v17 = (unsigned int)(v14 - 1);
    if ( (*(_BYTE *)(v16 + 23) & 0x40) == 0 )
      break;
    v18 = *(_QWORD *)(*(_QWORD *)(v16 - 8) + 24 * v17);
    if ( *(_BYTE *)(v18 + 16) != 13 )
      goto LABEL_27;
LABEL_17:
    v19 = *(_DWORD *)(v18 + 32);
    if ( v19 <= 0x40 )
    {
      v21 = *(_QWORD *)(v18 + 24) == 0;
    }
    else
    {
      v74 = v16;
      v20 = sub_16A57B0(v18 + 24);
      v16 = v74;
      v21 = v19 == v20;
    }
    v22 = v15;
    v23 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    v24 = v23;
    v25 = (v22 >> 2) & 1;
    if ( v21 )
      goto LABEL_33;
    if ( !(_BYTE)v25 )
    {
      if ( v23 )
      {
        v76 = v16;
        v38 = sub_15A9930(a4, v23);
        v16 = v76;
        v39 = *(_QWORD **)(v18 + 24);
        if ( *(_DWORD *)(v18 + 32) > 0x40u )
          v39 = (_QWORD *)*v39;
        v85 += *(_QWORD *)(v38 + 8LL * (unsigned int)v39 + 16);
        goto LABEL_42;
      }
LABEL_22:
      v75 = v16;
      v26 = sub_1643D30(0, *v84);
      v16 = v75;
      v27 = v26;
      goto LABEL_23;
    }
    v27 = v23;
    if ( !v23 )
      goto LABEL_22;
LABEL_23:
    v68 = v16;
    v28 = sub_15A9FE0(a4, v27);
    v16 = v68;
    v29 = 1;
    v30 = v28;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v27 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v48 = *(_QWORD *)(v27 + 32);
          v27 = *(_QWORD *)(v27 + 24);
          v29 *= v48;
          continue;
        case 1:
          v32 = 16;
          goto LABEL_30;
        case 2:
          v32 = 32;
          goto LABEL_30;
        case 3:
        case 9:
          v32 = 64;
          goto LABEL_30;
        case 4:
          v32 = 80;
          goto LABEL_30;
        case 5:
        case 6:
          v32 = 128;
          goto LABEL_30;
        case 7:
          v63 = v30;
          v41 = 0;
          v69 = v29;
          v78 = v16;
          goto LABEL_53;
        case 0xB:
          v32 = *(_DWORD *)(v27 + 8) >> 8;
          goto LABEL_30;
        case 0xD:
          v65 = v30;
          v71 = v29;
          v80 = v16;
          v47 = (_QWORD *)sub_15A9930(a4, v27);
          v16 = v80;
          v29 = v71;
          v30 = v65;
          v32 = 8LL * *v47;
          goto LABEL_30;
        case 0xE:
          v57 = v30;
          v60 = v29;
          v64 = v68;
          v70 = *(_QWORD *)(v27 + 24);
          v79 = *(_QWORD *)(v27 + 32);
          v43 = sub_15A9FE0(a4, v70);
          v30 = v57;
          v44 = v70;
          v45 = 1;
          v29 = v60;
          v16 = v64;
          v46 = v43;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v44 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v51 = *(_QWORD *)(v44 + 32);
                v44 = *(_QWORD *)(v44 + 24);
                v45 *= v51;
                continue;
              case 1:
                v49 = 16;
                goto LABEL_65;
              case 2:
                v49 = 32;
                goto LABEL_65;
              case 3:
              case 9:
                v49 = 64;
                goto LABEL_65;
              case 4:
                v49 = 80;
                goto LABEL_65;
              case 5:
              case 6:
                v49 = 128;
                goto LABEL_65;
              case 7:
                JUMPOUT(0x19CFE1F);
              case 0xB:
                v49 = *(_DWORD *)(v44 + 8) >> 8;
                goto LABEL_65;
              case 0xD:
                v56 = v57;
                v59 = v60;
                v62 = v46;
                v67 = v45;
                v73 = v16;
                v49 = 8LL * *(_QWORD *)sub_15A9930(a4, v44);
                goto LABEL_70;
              case 0xE:
                v52 = v57;
                v53 = v60;
                v54 = v46;
                v55 = v45;
                v58 = v64;
                v72 = *(_QWORD *)(v44 + 32);
                v61 = *(_QWORD *)(v44 + 24);
                v66 = (unsigned int)sub_15A9FE0(a4, v61);
                v50 = sub_127FA20(a4, v61);
                v16 = v58;
                v45 = v55;
                v46 = v54;
                v29 = v53;
                v30 = v52;
                v49 = 8 * v66 * v72 * ((v66 + ((unsigned __int64)(v50 + 7) >> 3) - 1) / v66);
                goto LABEL_65;
              case 0xF:
                v56 = v57;
                v59 = v60;
                v62 = v46;
                v67 = v45;
                v73 = v16;
                v49 = 8 * (unsigned int)sub_15A9520(a4, *(_DWORD *)(v44 + 8) >> 8);
LABEL_70:
                v16 = v73;
                v45 = v67;
                v46 = v62;
                v29 = v59;
                v30 = v56;
LABEL_65:
                v32 = 8 * v79 * v46 * ((v46 + ((unsigned __int64)(v45 * v49 + 7) >> 3) - 1) / v46);
                break;
            }
            goto LABEL_30;
          }
        case 0xF:
          v63 = v30;
          v69 = v29;
          v78 = v16;
          v41 = *(_DWORD *)(v27 + 8) >> 8;
LABEL_53:
          v42 = sub_15A9520(a4, v41);
          v16 = v78;
          v29 = v69;
          v30 = v63;
          v32 = (unsigned int)(8 * v42);
LABEL_30:
          v33 = *(_DWORD *)(v18 + 32);
          v34 = *(__int64 **)(v18 + 24);
          v35 = v30 * ((v30 + ((unsigned __int64)(v32 * v29 + 7) >> 3) - 1) / v30);
          if ( v33 > 0x40 )
            v36 = *v34;
          else
            v36 = (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
          v85 += v35 * v36;
          break;
      }
      break;
    }
LABEL_33:
    if ( (_BYTE)v25 && v23 )
    {
      v37 = *(_BYTE *)(v23 + 8);
      if ( ((v37 - 14) & 0xFD) == 0 )
        goto LABEL_36;
      goto LABEL_43;
    }
LABEL_42:
    v77 = v16;
    v40 = sub_1643D30(v23, *v84);
    v16 = v77;
    v24 = v40;
    v37 = *(_BYTE *)(v40 + 8);
    if ( ((v37 - 14) & 0xFD) == 0 )
    {
LABEL_36:
      v15 = *(_QWORD *)(v24 + 24) | 4LL;
      goto LABEL_37;
    }
LABEL_43:
    v15 = 0;
    if ( v37 == 13 )
      v15 = v24;
LABEL_37:
    v84 += 3;
    if ( v83 == v14 )
      return v85;
    ++v14;
  }
  v18 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) + 24 * v17);
  if ( *(_BYTE *)(v18 + 16) == 13 )
    goto LABEL_17;
LABEL_27:
  v85 = 1;
  *a3 = 1;
  return v85;
}
