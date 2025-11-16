// Function: sub_29D9730
// Address: 0x29d9730
//
__int64 __fastcall sub_29D9730(__int64 *a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r8
  int v9; // r9d
  int v10; // edx
  char v11; // si
  char v12; // cl
  bool v13; // al
  unsigned int v15; // r9d
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r11
  unsigned __int64 v19; // rax
  unsigned int v20; // r11d
  unsigned int v21; // r10d
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  bool v26; // al
  bool v27; // al
  bool v28; // al
  unsigned int v29; // r8d
  bool v30; // al
  bool v31; // al
  unsigned int v32; // r8d
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // rcx
  int v39; // eax
  const void *v40; // r13
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rbx
  const void *v43; // rax
  unsigned __int64 v44; // rdx
  bool v45; // al
  unsigned __int64 v46; // rsi
  __int64 v47; // rax
  unsigned __int64 v48; // rbx
  unsigned int v49; // eax
  __int64 v50; // r15
  unsigned int v51; // eax
  unsigned __int64 v52; // rbx
  unsigned int v53; // eax
  __int64 v54; // r15
  unsigned int v55; // eax
  unsigned int v56; // ebx
  unsigned int v57; // eax
  __int64 v58; // r15
  unsigned int v59; // eax
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rsi
  __int64 v64; // r8
  unsigned __int64 v65; // rbx
  unsigned int v66; // eax
  __int64 v67; // r15
  unsigned int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rdx
  unsigned int v72; // [rsp+8h] [rbp-A8h]
  int v73; // [rsp+10h] [rbp-A0h]
  unsigned int v74; // [rsp+14h] [rbp-9Ch]
  int v75; // [rsp+14h] [rbp-9Ch]
  char v76; // [rsp+1Bh] [rbp-95h]
  char v77; // [rsp+1Bh] [rbp-95h]
  unsigned int v78; // [rsp+1Ch] [rbp-94h]
  int v79; // [rsp+1Ch] [rbp-94h]
  unsigned int v80; // [rsp+1Ch] [rbp-94h]
  unsigned int v81; // [rsp+1Ch] [rbp-94h]
  unsigned int v82; // [rsp+1Ch] [rbp-94h]
  int v83; // [rsp+1Ch] [rbp-94h]
  int v84; // [rsp+1Ch] [rbp-94h]
  _BYTE v85[32]; // [rsp+20h] [rbp-90h] BYREF
  char v86; // [rsp+40h] [rbp-70h]
  unsigned __int64 v87; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v88; // [rsp+58h] [rbp-58h]
  char v89; // [rsp+70h] [rbp-40h]

  v6 = *((_QWORD *)a2 + 1);
  v7 = *((_QWORD *)a3 + 1);
  LODWORD(v8) = sub_29D81B0(a1, v6, v7);
  if ( (_DWORD)v8 )
  {
    v9 = *(unsigned __int8 *)(v7 + 8);
    v10 = *(unsigned __int8 *)(v6 + 8);
    v11 = *(_BYTE *)(v6 + 8);
    v12 = *(_BYTE *)(v7 + 8);
    v13 = (_BYTE)v9 != 13 && (_BYTE)v9 != 7;
    if ( (_BYTE)v10 == 7 || v10 == 13 )
    {
      if ( !v13 )
        return (unsigned int)v8;
      goto LABEL_26;
    }
    if ( !v13 )
    {
LABEL_5:
      LODWORD(v8) = 1;
      return (unsigned int)v8;
    }
    v15 = v9 - 17;
    if ( (unsigned int)(v10 - 17) > 1 )
    {
      if ( v15 > 1 )
        goto LABEL_31;
      v75 = v8;
      v77 = *(_BYTE *)(v7 + 8);
      v47 = sub_BCAE30(v7);
      v21 = 0;
      v12 = v77;
      v87 = v47;
      LODWORD(v8) = v75;
      v88 = v22;
      LODWORD(v22) = v47;
      if ( !(_DWORD)v47 )
        goto LABEL_31;
    }
    else
    {
      v73 = v8;
      v74 = v15;
      v76 = *(_BYTE *)(v7 + 8);
      v16 = sub_BCAE30(v6);
      v18 = v17;
      v22 = v16;
      v12 = v76;
      LODWORD(v8) = v73;
      v19 = v18;
      v87 = v22;
      v20 = v22;
      v21 = v22;
      LODWORD(v22) = 0;
      v88 = v19;
      if ( v74 <= 1 )
      {
        v72 = v20;
        v78 = v20;
        v23 = sub_BCAE30(v7);
        v20 = v72;
        LODWORD(v8) = v73;
        v24 = v23;
        v25 = v22;
        v12 = v76;
        v87 = v24;
        v21 = v78;
        LODWORD(v22) = v24;
        v88 = v25;
      }
      if ( (_DWORD)v22 == v20 )
      {
        if ( (_DWORD)v22 )
          goto LABEL_13;
LABEL_31:
        if ( v11 == 14 )
        {
          if ( v12 == 14 )
          {
            LODWORD(v8) = sub_29D7CF0((__int64)a1, *(_DWORD *)(v6 + 8) >> 8, *(_DWORD *)(v7 + 8) >> 8);
            if ( (_DWORD)v8 )
              return (unsigned int)v8;
          }
          goto LABEL_5;
        }
        if ( v12 != 14 )
          return (unsigned int)v8;
LABEL_26:
        LODWORD(v8) = -1;
        return (unsigned int)v8;
      }
    }
    v22 = (unsigned int)v22;
    v46 = v21;
    return sub_29D7CF0((__int64)a1, v46, v22);
  }
LABEL_13:
  v79 = v8;
  v26 = sub_AC30F0((__int64)a2);
  LODWORD(v8) = v79;
  if ( v26 )
  {
    v27 = sub_AC30F0((__int64)a3);
    LODWORD(v8) = v79;
    if ( v27 )
      return (unsigned int)v8;
  }
  v80 = v8;
  v28 = sub_AC30F0((__int64)a2);
  v29 = v80;
  if ( v28 )
  {
    v30 = sub_AC30F0((__int64)a3);
    v29 = v80;
    if ( !v30 )
      goto LABEL_5;
  }
  v81 = v29;
  v31 = sub_AC30F0((__int64)a2);
  v32 = v81;
  if ( !v31 )
  {
    v45 = sub_AC30F0((__int64)a3);
    v32 = v81;
    if ( v45 )
      goto LABEL_26;
  }
  v33 = *a2;
  v34 = *a3;
  if ( (unsigned __int8)v33 > 3u || (unsigned __int8)v34 > 3u )
  {
    v82 = v32;
    v37 = sub_29D7CF0((__int64)a1, v33, v34);
    if ( v37 )
    {
      LODWORD(v8) = v37;
      return (unsigned int)v8;
    }
    LODWORD(v8) = v82;
    v39 = *a2;
    if ( (unsigned int)(v39 - 15) <= 1 )
    {
      v40 = (const void *)sub_AC52D0((__int64)a3);
      v42 = v41;
      v43 = (const void *)sub_AC52D0((__int64)a2);
      return sub_29D7F50((__int64)a1, v43, v44, v40, v42);
    }
    switch ( (char)v39 )
    {
      case 4:
        LODWORD(v8) = sub_29DA390(a1, *((_QWORD *)a2 - 8), *((_QWORD *)a3 - 8), v38, v82);
        if ( (_DWORD)v8 )
          return (unsigned int)v8;
        v62 = *((_QWORD *)a2 - 8);
        v63 = *((_QWORD *)a2 - 4);
        v64 = *((_QWORD *)a3 - 4);
        if ( *((_QWORD *)a3 - 8) != v62 )
          return sub_29DA390(a1, v63, *((_QWORD *)a3 - 4), v62, v64);
        if ( v63 == v64 )
          goto LABEL_43;
        v69 = *(_QWORD *)(v62 + 80);
        v70 = v62 + 72;
        if ( v70 == v69 )
LABEL_97:
          BUG();
        while ( 1 )
        {
          v71 = v69 - 24;
          if ( !v69 )
            v71 = 0;
          if ( v71 == v63 )
            goto LABEL_26;
          if ( v71 == v64 )
            goto LABEL_5;
          v69 = *(_QWORD *)(v69 + 8);
          if ( v70 == v69 )
            goto LABEL_97;
        }
      case 5:
        LODWORD(v8) = sub_29D7CF0((__int64)a1, *((unsigned __int16 *)a2 + 1), *((unsigned __int16 *)a3 + 1));
        if ( (_DWORD)v8 )
          return (unsigned int)v8;
        v56 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
        v57 = sub_29D7CF0((__int64)a1, v56, *((_DWORD *)a3 + 1) & 0x7FFFFFF);
        v8 = v57;
        if ( v57 )
          return (unsigned int)v8;
        v58 = 0;
        if ( !v56 )
          goto LABEL_56;
        do
        {
          v59 = sub_29D9730(
                  a1,
                  *(_QWORD *)&a2[32 * (v58 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                  *(_QWORD *)&a3[32 * (v58 - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))],
                  *((_DWORD *)a2 + 1) & 0x7FFFFFF,
                  v8);
          v8 = v59;
          if ( v59 )
            return (unsigned int)v8;
          ++v58;
        }
        while ( v56 > (unsigned int)v58 );
LABEL_56:
        if ( *((_WORD *)a2 + 1) != 34 )
          goto LABEL_66;
        v60 = sub_BB5290((__int64)a3);
        v61 = sub_BB5290((__int64)a2);
        LODWORD(v8) = sub_29D81B0(a1, v61, v60);
        if ( (_DWORD)v8 )
          return (unsigned int)v8;
        LODWORD(v8) = sub_29D7CF0((__int64)a1, a2[1] >> 1, a3[1] >> 1);
        if ( (_DWORD)v8 )
          return (unsigned int)v8;
        sub_BB52D0((__int64)v85, (__int64)a2);
        sub_BB52D0((__int64)&v87, (__int64)a3);
        if ( v86 )
        {
          LODWORD(v8) = 1;
          if ( !v89 )
            goto LABEL_93;
          LODWORD(v8) = sub_29D7DA0((__int64)a1, (__int64)v85, (__int64)&v87);
          if ( !(_DWORD)v8 )
          {
            if ( v89 )
              sub_9963D0((__int64)&v87);
LABEL_64:
            if ( v86 )
              sub_9963D0((__int64)v85);
LABEL_66:
            if ( (*((_WORD *)a2 + 1) & 0xFFF7) != 0x11 && (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD )
              goto LABEL_43;
            LODWORD(v8) = sub_29D7CF0((__int64)a1, (a2[1] & 2) != 0, (a3[1] & 2) != 0);
            if ( (_DWORD)v8 )
              return (unsigned int)v8;
            v22 = (a3[1] & 4) != 0;
            v46 = (a2[1] & 4) != 0;
            return sub_29D7CF0((__int64)a1, v46, v22);
          }
          if ( !v89 )
          {
LABEL_93:
            if ( v86 )
            {
              v84 = v8;
              sub_9963D0((__int64)v85);
              LODWORD(v8) = v84;
            }
            return (unsigned int)v8;
          }
        }
        else
        {
          if ( !v89 )
            goto LABEL_64;
          LODWORD(v8) = -1;
        }
        v83 = v8;
        sub_9963D0((__int64)&v87);
        LODWORD(v8) = v83;
        goto LABEL_93;
      case 6:
        v35 = *((_QWORD *)a3 - 4);
        v36 = *((_QWORD *)a2 - 4);
        return sub_29D96E0((__int64)a1, v36, v35);
      case 9:
        v65 = *(_QWORD *)(v6 + 32);
        v66 = sub_29D7CF0((__int64)a1, v65, *(_QWORD *)(v7 + 32));
        v8 = v66;
        if ( v66 )
          return (unsigned int)v8;
        v67 = 0;
        if ( !v65 )
          goto LABEL_43;
        while ( 1 )
        {
          v68 = sub_29D9730(
                  a1,
                  *(_QWORD *)&a2[32 * ((unsigned int)v67 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                  *(_QWORD *)&a3[32 * ((unsigned int)v67 - (unsigned __int64)(*((_DWORD *)a3 + 1) & 0x7FFFFFF))],
                  *((_DWORD *)a2 + 1) & 0x7FFFFFF,
                  v8);
          v8 = v68;
          if ( v68 )
            return (unsigned int)v8;
          if ( v65 == ++v67 )
            goto LABEL_43;
        }
      case 10:
        v52 = *(unsigned int *)(v6 + 12);
        v53 = sub_29D7CF0((__int64)a1, v52, *(unsigned int *)(v7 + 12));
        v8 = v53;
        if ( v53 )
          return (unsigned int)v8;
        if ( !(_DWORD)v52 )
          goto LABEL_43;
        v54 = 0;
        while ( 1 )
        {
          v55 = sub_29D9730(
                  a1,
                  *(_QWORD *)&a2[32 * (v54 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                  *(_QWORD *)&a3[32 * (v54 - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))],
                  *((_DWORD *)a2 + 1) & 0x7FFFFFF,
                  v8);
          v8 = v55;
          if ( v55 )
            return (unsigned int)v8;
          if ( ++v54 == v52 )
            goto LABEL_43;
        }
      case 11:
        v48 = *(unsigned int *)(v6 + 32);
        v49 = sub_29D7CF0((__int64)a1, v48, *(unsigned int *)(v7 + 32));
        v8 = v49;
        if ( v49 )
          return (unsigned int)v8;
        if ( !v48 )
          goto LABEL_43;
        v50 = 0;
        while ( 1 )
        {
          v51 = sub_29D9730(
                  a1,
                  *(_QWORD *)&a2[32 * (v50 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                  *(_QWORD *)&a3[32 * (v50 - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))],
                  *((_DWORD *)a2 + 1) & 0x7FFFFFF,
                  v8);
          v8 = v51;
          if ( v51 )
            break;
          if ( v48 == ++v50 )
          {
LABEL_43:
            LODWORD(v8) = 0;
            return (unsigned int)v8;
          }
        }
        return (unsigned int)v8;
      case 12:
      case 13:
      case 21:
        return (unsigned int)v8;
      case 17:
        return sub_29D7D50((__int64)a1, (__int64)(a2 + 24), (__int64)(a3 + 24));
      case 18:
        return sub_29D7DF0((__int64)a1, (__int64 *)a2 + 3, (__int64 *)a3 + 3);
      default:
        goto LABEL_97;
    }
  }
  v35 = (__int64)a3;
  v36 = (__int64)a2;
  return sub_29D96E0((__int64)a1, v36, v35);
}
