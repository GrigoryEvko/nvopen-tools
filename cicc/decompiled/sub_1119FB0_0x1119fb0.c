// Function: sub_1119FB0
// Address: 0x1119fb0
//
_QWORD *__fastcall sub_1119FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  int v7; // r13d
  unsigned int v8; // edi
  bool v9; // al
  char v10; // al
  __int64 v11; // rcx
  _QWORD *v12; // r14
  __int64 v14; // rdx
  _BYTE *v15; // rax
  bool v16; // al
  __int64 *v17; // rsi
  unsigned int **v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // eax
  _BYTE *v23; // rax
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned int v28; // eax
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  _QWORD *v32; // rax
  unsigned int v33; // ebx
  int v34; // eax
  __int64 v35; // rax
  _QWORD *v36; // rax
  unsigned __int8 *v37; // rax
  __int64 v38; // r12
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  unsigned int v41; // eax
  _QWORD *v42; // rax
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  unsigned int v45; // eax
  unsigned int v46; // edx
  __int64 v47; // rax
  unsigned int **v48; // r14
  unsigned int v49; // eax
  _BYTE *v50; // rax
  __int64 v51; // r13
  _QWORD *v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned int **v55; // r12
  _BYTE *v56; // rax
  __int64 v57; // r12
  _QWORD *v58; // rax
  __int64 v59; // [rsp+10h] [rbp-110h]
  char v60; // [rsp+10h] [rbp-110h]
  char v61; // [rsp+10h] [rbp-110h]
  __int16 v62; // [rsp+18h] [rbp-108h]
  __int64 v63; // [rsp+20h] [rbp-100h]
  __int64 v64; // [rsp+28h] [rbp-F8h]
  __int16 v65; // [rsp+30h] [rbp-F0h]
  char v66; // [rsp+36h] [rbp-EAh]
  bool v67; // [rsp+37h] [rbp-E9h]
  __int64 v69; // [rsp+38h] [rbp-E8h]
  __int64 *v70; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v71; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v72; // [rsp+58h] [rbp-C8h]
  __int64 v73; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v74; // [rsp+68h] [rbp-B8h]
  __int64 v75; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v76; // [rsp+78h] [rbp-A8h]
  __int64 v77; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v78; // [rsp+88h] [rbp-98h]
  __int64 v79; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v80; // [rsp+98h] [rbp-88h]
  __int16 v81; // [rsp+B0h] [rbp-70h]
  const char *v82; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v83; // [rsp+C8h] [rbp-58h]
  __int16 v84; // [rsp+E0h] [rbp-40h]

  v6 = *(_QWORD *)(a3 - 64);
  v64 = *(_QWORD *)(a3 - 32);
  v7 = *(_WORD *)(a2 + 2) & 0x3F;
  v8 = v7;
  v62 = *(_WORD *)(a2 + 2) & 0x3F;
  v63 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)(v7 - 32) > 1 || *(_BYTE *)v6 > 0x15u || *(_BYTE *)v6 == 5 )
  {
LABEL_2:
    v72 = 1;
    v71 = 0;
    v65 = sub_B52F50(v8);
    v66 = sub_B44900(a3);
    v67 = sub_B448F0(a3);
    if ( *(_BYTE *)v6 == 17 )
    {
      v70 = (__int64 *)(v6 + 24);
    }
    else
    {
      v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
      if ( (unsigned int)v14 > 1 )
        goto LABEL_7;
      if ( *(_BYTE *)v6 > 0x15u )
        goto LABEL_7;
      v15 = sub_AD7630(v6, 0, v14);
      if ( !v15 || *v15 != 17 )
        goto LABEL_7;
      v70 = (__int64 *)(v15 + 24);
    }
    v9 = sub_B532A0(*(_WORD *)(a2 + 2) & 0x3F);
    if ( !v67 || !v9 )
    {
      v16 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
      if ( !v66 )
        goto LABEL_7;
      if ( !v16 )
      {
        if ( !(unsigned __int8)sub_F0C3D0(a1) )
        {
LABEL_8:
          v11 = *(_QWORD *)(a3 + 16);
          if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
            goto LABEL_9;
LABEL_42:
          v33 = *(_DWORD *)(a4 + 8);
          if ( v33 <= 0x40 )
          {
            if ( !*(_QWORD *)a4 )
              goto LABEL_44;
          }
          else
          {
            v59 = v11;
            v34 = sub_C444A0(a4);
            v11 = v59;
            if ( v33 == v34 )
            {
LABEL_44:
              if ( !v11 )
              {
LABEL_48:
                v84 = 257;
                v36 = sub_BD2C40(72, unk_3F10FD0);
                v12 = v36;
                if ( v36 )
                  sub_1113300((__int64)v36, v7, v6, v64, (__int64)&v82);
                goto LABEL_11;
              }
              v35 = v11;
              while ( **(_BYTE **)(v35 + 24) != 84 )
              {
                v35 = *(_QWORD *)(v35 + 8);
                if ( !v35 )
                  goto LABEL_48;
              }
              goto LABEL_26;
            }
          }
LABEL_9:
          if ( !v11 )
          {
LABEL_10:
            v12 = 0;
LABEL_11:
            if ( v72 > 0x40 && v71 )
              j_j___libc_free_0_0(v71);
            return v12;
          }
LABEL_26:
          if ( *(_QWORD *)(v11 + 8) )
            goto LABEL_10;
          if ( !sub_B44900(a3) )
            goto LABEL_30;
          if ( v62 == 38 )
          {
            if ( sub_986760(a4) )
            {
              v84 = 257;
              v42 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v42;
              if ( v42 )
                sub_1113300((__int64)v42, 39, v6, v64, (__int64)&v82);
              goto LABEL_11;
            }
            if ( sub_9867B0(a4) )
            {
              v84 = 257;
              v43 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v43;
              if ( v43 )
                sub_1113300((__int64)v43, 38, v6, v64, (__int64)&v82);
              goto LABEL_11;
            }
          }
          else
          {
            if ( v62 != 40 )
            {
LABEL_30:
              LOBYTE(v83) = 0;
              v82 = (const char *)&v70;
              if ( !(unsigned __int8)sub_991580((__int64)&v82, v6) )
                goto LABEL_10;
              if ( v62 == 36 )
              {
                if ( sub_986BA0(a4) )
                {
                  sub_9865C0((__int64)&v79, a4);
                  sub_C46F20((__int64)&v79, 1u);
                  v45 = v80;
                  v80 = 0;
                  v83 = v45;
                  v82 = (const char *)v79;
                  sub_9865C0((__int64)&v73, a4);
                  sub_C46F20((__int64)&v73, 1u);
                  v46 = v74;
                  v74 = 0;
                  v76 = v46;
                  v75 = v73;
                  if ( v46 > 0x40 )
                  {
                    sub_C43B90(&v75, v70);
                    v46 = v76;
                    v47 = v75;
                  }
                  else
                  {
                    v47 = *v70 & v73;
                    v75 = v47;
                  }
                  v78 = v46;
                  v77 = v47;
                  v76 = 0;
                  v60 = sub_AAD8B0((__int64)&v77, &v82);
                  sub_969240(&v77);
                  sub_969240(&v75);
                  sub_969240(&v73);
                  sub_969240((__int64 *)&v82);
                  sub_969240(&v79);
                  if ( v60 )
                  {
                    v48 = *(unsigned int ***)(a1 + 32);
                    v81 = 257;
                    sub_9865C0((__int64)&v75, a4);
                    sub_C46F20((__int64)&v75, 1u);
                    v49 = v76;
                    v76 = 0;
                    v78 = v49;
                    v77 = v75;
                    v50 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v64 + 8), (__int64)&v77);
                    v51 = sub_A82480(v48, (_BYTE *)v64, v50, (__int64)&v79);
                    v84 = 257;
                    v52 = sub_BD2C40(72, unk_3F10FD0);
                    v12 = v52;
                    if ( v52 )
                      sub_1113300((__int64)v52, 32, v51, v6, (__int64)&v82);
                    sub_969240(&v77);
                    sub_969240(&v75);
                    goto LABEL_11;
                  }
                }
              }
              else if ( v62 == 34 )
              {
                sub_9865C0((__int64)&v75, a4);
                sub_C46A40((__int64)&v75, 1);
                v41 = v76;
                v76 = 0;
                v78 = v41;
                v77 = v75;
                if ( sub_986BA0((__int64)&v77) )
                {
                  sub_9865C0((__int64)&v79, (__int64)v70);
                  v53 = v80;
                  if ( v80 > 0x40 )
                  {
                    sub_C43B90(&v79, (__int64 *)a4);
                    v53 = v80;
                    v54 = v79;
                  }
                  else
                  {
                    v54 = *(_QWORD *)a4 & v79;
                    v79 = v54;
                  }
                  v83 = v53;
                  v82 = (const char *)v54;
                  v80 = 0;
                  v61 = sub_AAD8B0((__int64)&v82, (_QWORD *)a4);
                  sub_969240((__int64 *)&v82);
                  sub_969240(&v79);
                  sub_969240(&v77);
                  sub_969240(&v75);
                  if ( v61 )
                  {
                    v55 = *(unsigned int ***)(a1 + 32);
                    v81 = 257;
                    v56 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v64 + 8), a4);
                    v57 = sub_A82480(v55, (_BYTE *)v64, v56, (__int64)&v79);
                    v84 = 257;
                    v58 = sub_BD2C40(72, unk_3F10FD0);
                    v12 = v58;
                    if ( v58 )
                      sub_1113300((__int64)v58, 33, v57, v6, (__int64)&v82);
                    goto LABEL_11;
                  }
                }
                else
                {
                  sub_969240(&v77);
                  sub_969240(&v75);
                }
              }
LABEL_33:
              v17 = v70;
              v18 = *(unsigned int ***)(a1 + 32);
              v84 = 259;
              v82 = "notsub";
              sub_9865C0((__int64)&v77, (__int64)v70);
              sub_987160((__int64)&v77, (__int64)v17, v19, v20, v21);
              v22 = v78;
              v78 = 0;
              v80 = v22;
              v79 = v77;
              v23 = (_BYTE *)sub_AD8D80(v63, (__int64)&v79);
              v24 = sub_929C50(v18, (_BYTE *)v64, v23, (__int64)&v82, v67, v66);
              sub_969240(&v79);
              sub_969240(&v77);
              sub_9865C0((__int64)&v77, a4);
              sub_987160((__int64)&v77, a4, v25, v26, v27);
              v28 = v78;
              v78 = 0;
              v80 = v28;
              v79 = v77;
              v69 = sub_AD8D80(v63, (__int64)&v79);
              v84 = 257;
              v29 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v29;
              if ( v29 )
                sub_1113300((__int64)v29, v65, v24, v69, (__int64)&v82);
              sub_969240(&v79);
              sub_969240(&v77);
              goto LABEL_11;
            }
            if ( sub_9867B0(a4) )
            {
              v84 = 257;
              v44 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v44;
              if ( v44 )
                sub_1113300((__int64)v44, 40, v6, v64, (__int64)&v82);
              goto LABEL_11;
            }
            if ( sub_D94040(a4) )
            {
              v84 = 257;
              v40 = sub_BD2C40(72, unk_3F10FD0);
              v12 = v40;
              if ( v40 )
                sub_1113300((__int64)v40, 41, v6, v64, (__int64)&v82);
              goto LABEL_11;
            }
          }
          LOBYTE(v83) = 0;
          v82 = (const char *)&v70;
          if ( !(unsigned __int8)sub_991580((__int64)&v82, v6) )
            goto LABEL_10;
          goto LABEL_33;
        }
LABEL_23:
        v11 = *(_QWORD *)(a3 + 16);
        if ( !v11 || *(_QWORD *)(v11 + 8) )
          goto LABEL_10;
        if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
          goto LABEL_26;
        goto LABEL_42;
      }
    }
    v10 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
    if ( !sub_1110FE0(&v71, (__int64)v70, (__int64 *)a4, v10) )
    {
      v30 = sub_AD8D80(v63, (__int64)&v71);
      v84 = 257;
      v31 = v30;
      v32 = sub_BD2C40(72, unk_3F10FD0);
      v12 = v32;
      if ( v32 )
        sub_1113300((__int64)v32, v65, v64, v31, (__int64)&v82);
      goto LABEL_11;
    }
LABEL_7:
    if ( !(unsigned __int8)sub_F0C3D0(a1) )
      goto LABEL_8;
    goto LABEL_23;
  }
  if ( (unsigned __int8)sub_AD6CA0(v6) )
  {
    v8 = *(_WORD *)(a2 + 2) & 0x3F;
    goto LABEL_2;
  }
  v37 = (unsigned __int8 *)sub_AD8D80(v63, a4);
  v84 = 257;
  v38 = sub_AD57F0(v6, v37, 0, 0);
  v39 = sub_BD2C40(72, unk_3F10FD0);
  v12 = v39;
  if ( v39 )
    sub_1113300((__int64)v39, v7, v64, v38, (__int64)&v82);
  return v12;
}
