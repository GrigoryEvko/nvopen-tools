// Function: sub_1285550
// Address: 0x1285550
//
__int64 __fastcall sub_1285550(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r14
  unsigned int v16; // ecx
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rdi
  _BOOL4 v26; // edx
  int v27; // ebx
  unsigned __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdi
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // rdi
  _BOOL4 v38; // edx
  int v39; // ebx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r11
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned __int64 *v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // rcx
  __int64 v52; // rsi
  _QWORD *v53; // rdx
  __int64 v54; // rsi
  __int64 v55; // rax
  _QWORD *v56; // rdi
  __int64 v57; // rax
  __int64 v58; // r15
  unsigned int v59; // ebx
  __int64 v60; // rax
  _QWORD *v61; // r13
  __int64 v62; // rax
  unsigned __int64 *v63; // rbx
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  __int64 v66; // rsi
  __int64 v67; // rsi
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // [rsp+8h] [rbp-E8h]
  unsigned int v71; // [rsp+10h] [rbp-E0h]
  __int64 v72; // [rsp+18h] [rbp-D8h]
  char *v73; // [rsp+20h] [rbp-D0h]
  __int64 *v74; // [rsp+30h] [rbp-C0h]
  __int64 v75; // [rsp+38h] [rbp-B8h]
  unsigned int v76; // [rsp+40h] [rbp-B0h]
  __int64 v77; // [rsp+48h] [rbp-A8h]
  unsigned __int64 *v78; // [rsp+48h] [rbp-A8h]
  _BYTE *v80; // [rsp+58h] [rbp-98h]
  int v81; // [rsp+64h] [rbp-8Ch] BYREF
  __int64 v82; // [rsp+68h] [rbp-88h] BYREF
  __int64 v83; // [rsp+70h] [rbp-80h] BYREF
  __int64 v84; // [rsp+78h] [rbp-78h]
  _QWORD v85[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v86; // [rsp+90h] [rbp-60h]
  _QWORD v87[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v88; // [rsp+B0h] [rbp-40h]

  v3 = a3;
  if ( (unsigned __int8)sub_127F7A0(a2[4], a3, &v81) )
  {
    v5 = *(_QWORD *)(v3 + 120);
    v87[0] = "predef_tmp";
    v6 = v81;
    v88 = 259;
    v80 = sub_127FDC0(a2, v5, (__int64)v87);
    v74 = a2 + 6;
    if ( (_DWORD)v6 == 4 )
    {
      v56 = (_QWORD *)a2[4];
      v88 = 257;
      v57 = sub_126A190(v56, 4348, 0, 0);
      v58 = sub_1285290(v74, *(_QWORD *)(*(_QWORD *)v57 + 24LL), v57, 0, 0, (__int64)v87, 0);
      v59 = unk_4D0463C;
      if ( unk_4D0463C )
        v59 = sub_126A420(a2[4], (unsigned __int64)v80);
      v88 = 257;
      v60 = sub_1648A60(64, 2);
      v61 = (_QWORD *)v60;
      if ( v60 )
        sub_15F9650(v60, v58, v80, v59, 0);
      v62 = a2[7];
      if ( v62 )
      {
        v63 = (unsigned __int64 *)a2[8];
        sub_157E9D0(v62 + 40, v61);
        v64 = v61[3];
        v65 = *v63;
        v61[4] = v63;
        v65 &= 0xFFFFFFFFFFFFFFF8LL;
        v61[3] = v65 | v64 & 7;
        *(_QWORD *)(v65 + 8) = v61 + 3;
        *v63 = *v63 & 7 | (unsigned __int64)(v61 + 3);
      }
      sub_164B780(v61, v87);
      v66 = a2[6];
      if ( v66 )
      {
        v85[0] = a2[6];
        sub_1623A60(v85, v66, 2);
        if ( v61[6] )
          sub_161E7C0(v61 + 6);
        v67 = v85[0];
        v61[6] = v85[0];
        if ( v67 )
          sub_1623210(v85, v67, v61 + 6);
      }
      v68 = *(_QWORD *)(v3 + 120);
      if ( *(char *)(v68 + 142) >= 0 && *(_BYTE *)(v68 + 140) == 12 )
        v23 = (unsigned int)sub_8D4AB0(v68);
      else
        v23 = *(unsigned int *)(v68 + 136);
      sub_15F9450(v61, v23);
    }
    else
    {
      v7 = 3 * v6;
      v75 = v3;
      v8 = 0;
      v73 = (char *)&unk_427F760 + 4 * v7;
      do
      {
        v88 = 257;
        v9 = sub_126A190((_QWORD *)a2[4], *(unsigned int *)&v73[4 * v8], 0, 0);
        v77 = sub_1285290(v74, *(_QWORD *)(*(_QWORD *)v9 + 24LL), v9, 0, 0, (__int64)v87, 0);
        v86 = 257;
        v10 = sub_127A030(a2[4] + 8LL, *(_QWORD *)(v75 + 120), 0);
        v11 = sub_1643350(a2[9]);
        v12 = sub_159C470(v11, 0, 0);
        v13 = a2[9];
        v83 = v12;
        v14 = sub_1643350(v13);
        v84 = sub_159C470(v14, v8, 0);
        if ( v80[16] > 0x10u )
        {
          v88 = 257;
          if ( !v10 )
          {
            v69 = *(_QWORD *)v80;
            if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16 )
              v69 = **(_QWORD **)(v69 + 16);
            v10 = *(_QWORD *)(v69 + 24);
          }
          v40 = sub_1648A60(72, 3);
          v15 = (_QWORD *)v40;
          if ( v40 )
          {
            v72 = v40;
            v41 = v40 - 72;
            v42 = *(_QWORD *)v80;
            if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16 )
              v42 = **(_QWORD **)(v42 + 16);
            v70 = v41;
            v71 = *(_DWORD *)(v42 + 8) >> 8;
            v43 = sub_15F9F50(v10, &v83, 2);
            v44 = sub_1646BA0(v43, v71);
            v45 = v70;
            v46 = v44;
            v47 = *(_QWORD *)v80;
            if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16
              || (v47 = *(_QWORD *)v83, *(_BYTE *)(*(_QWORD *)v83 + 8LL) == 16)
              || (v47 = *(_QWORD *)v84, *(_BYTE *)(*(_QWORD *)v84 + 8LL) == 16) )
            {
              v55 = sub_16463B0(v46, *(_QWORD *)(v47 + 32));
              v45 = v70;
              v46 = v55;
            }
            sub_15F1EA0(v15, v46, 32, v45, 3, 0);
            v15[7] = v10;
            v15[8] = sub_15F9F50(v10, &v83, 2);
            sub_15F9CE0(v15, v80, &v83, 2, v87);
          }
          else
          {
            v72 = 0;
          }
          v48 = a2[7];
          if ( v48 )
          {
            v49 = (unsigned __int64 *)a2[8];
            sub_157E9D0(v48 + 40, v15);
            v50 = v15[3];
            v51 = *v49;
            v15[4] = v49;
            v51 &= 0xFFFFFFFFFFFFFFF8LL;
            v15[3] = v51 | v50 & 7;
            *(_QWORD *)(v51 + 8) = v15 + 3;
            *v49 = *v49 & 7 | (unsigned __int64)(v15 + 3);
          }
          sub_164B780(v72, v85);
          v52 = a2[6];
          if ( v52 )
          {
            v82 = a2[6];
            sub_1623A60(&v82, v52, 2);
            v53 = v15 + 6;
            if ( v15[6] )
            {
              sub_161E7C0(v15 + 6);
              v53 = v15 + 6;
            }
            v54 = v82;
            v15[6] = v82;
            if ( v54 )
              sub_1623210(&v82, v54, v53);
          }
        }
        else
        {
          BYTE4(v87[0]) = 0;
          v15 = (_QWORD *)sub_15A2E80(v10, (_DWORD)v80, (unsigned int)&v83, 2, 0, (unsigned int)v87, 0);
        }
        v16 = unk_4D0463C;
        if ( unk_4D0463C )
          v16 = sub_126A420(a2[4], (unsigned __int64)v15);
        v76 = v16;
        v88 = 257;
        v17 = sub_1648A60(64, 2);
        v18 = (_QWORD *)v17;
        if ( v17 )
          sub_15F9650(v17, v77, v15, v76, 0);
        v19 = a2[7];
        if ( v19 )
        {
          v78 = (unsigned __int64 *)a2[8];
          sub_157E9D0(v19 + 40, v18);
          v20 = *v78;
          v21 = v18[3] & 7LL;
          v18[4] = v78;
          v20 &= 0xFFFFFFFFFFFFFFF8LL;
          v18[3] = v20 | v21;
          *(_QWORD *)(v20 + 8) = v18 + 3;
          *v78 = *v78 & 7 | (unsigned __int64)(v18 + 3);
        }
        sub_164B780(v18, v87);
        v23 = a2[6];
        if ( v23 )
        {
          v85[0] = a2[6];
          sub_1623A60(v85, v23, 2);
          v22 = (__int64)(v18 + 6);
          if ( v18[6] )
          {
            sub_161E7C0(v18 + 6);
            v22 = (__int64)(v18 + 6);
          }
          v23 = v85[0];
          v18[6] = v85[0];
          if ( v23 )
            sub_1623210(v85, v23, v22);
        }
        ++v8;
      }
      while ( v8 != 3 );
      v3 = v75;
    }
    v24 = sub_127C800(v3, v23, v22);
    v25 = *(_QWORD *)(v3 + 120);
    v26 = 0;
    v27 = v24;
    if ( (*(_BYTE *)(v25 + 140) & 0xFB) == 8 )
      v26 = (sub_8D4C10(v25, dword_4F077C4 != 2) & 2) != 0;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v80;
    *(_DWORD *)(a1 + 40) = v26;
    *(_DWORD *)(a1 + 16) = v27;
  }
  else
  {
    v29 = sub_1277140((__int64 *)a2[4], v3, 0);
    v30 = a2[4];
    v31 = v30 + 544;
    v32 = *(_QWORD **)(v30 + 552);
    if ( !v32 )
      goto LABEL_49;
    v33 = v31;
    do
    {
      while ( 1 )
      {
        v34 = v32[2];
        v35 = v32[3];
        if ( v32[4] >= v29 )
          break;
        v32 = (_QWORD *)v32[3];
        if ( !v35 )
          goto LABEL_28;
      }
      v33 = (__int64)v32;
      v32 = (_QWORD *)v32[2];
    }
    while ( v34 );
LABEL_28:
    if ( v31 == v33 || *(_QWORD *)(v33 + 32) > v29 )
    {
LABEL_49:
      v33 = v29;
      v29 = sub_1289750(a2, v29, v3 + 64);
    }
    v36 = sub_127C800(v3, v33, v35);
    v37 = *(_QWORD *)(v3 + 120);
    v38 = 0;
    v39 = v36;
    if ( (*(_BYTE *)(v37 + 140) & 0xFB) == 8 )
      v38 = (sub_8D4C10(v37, dword_4F077C4 != 2) & 2) != 0;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v29;
    *(_DWORD *)(a1 + 40) = v38;
    *(_DWORD *)(a1 + 16) = v39;
  }
  return a1;
}
