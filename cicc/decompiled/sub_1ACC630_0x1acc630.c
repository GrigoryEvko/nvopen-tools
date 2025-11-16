// Function: sub_1ACC630
// Address: 0x1acc630
//
__int64 __fastcall sub_1ACC630(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // r8d
  char v12; // di
  bool v13; // al
  bool v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  bool v18; // al
  bool v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned int v22; // r8d
  bool v23; // al
  bool v24; // al
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned int v27; // r8d
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rcx
  int v32; // eax
  const void *v33; // r13
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rbx
  const void *v36; // rax
  unsigned __int64 v37; // rdx
  bool v38; // al
  int v39; // eax
  unsigned int v40; // eax
  unsigned __int64 v41; // rbx
  __int64 v42; // r15
  unsigned int v43; // ebx
  __int64 v44; // r15
  unsigned __int64 v45; // rbx
  __int64 v46; // r15
  unsigned __int64 v47; // rbx
  __int64 v48; // r15
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r8
  __int64 v53; // rdx
  __int64 v54; // rax
  unsigned int v55; // [rsp+0h] [rbp-40h]
  unsigned int v56; // [rsp+4h] [rbp-3Ch]
  unsigned __int8 v57; // [rsp+4h] [rbp-3Ch]
  unsigned __int8 v58; // [rsp+Bh] [rbp-35h]
  unsigned __int8 v59; // [rsp+Bh] [rbp-35h]
  unsigned int v60; // [rsp+Ch] [rbp-34h]
  unsigned int v61; // [rsp+Ch] [rbp-34h]
  unsigned int v62; // [rsp+Ch] [rbp-34h]
  unsigned int v63; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)a2;
  v7 = *(_QWORD *)a3;
  v8 = *(_QWORD *)a2;
  v11 = sub_1ACB220(a1, v8, *(_QWORD *)a3);
  if ( !v11 )
    goto LABEL_9;
  LOBYTE(v10) = *(_BYTE *)(v7 + 8);
  v12 = *(_BYTE *)(v6 + 8);
  LOBYTE(v9) = v12;
  v13 = (_BYTE)v10 != 12 && (_BYTE)v10 != 0;
  if ( v12 == 12 || !*(_BYTE *)(v6 + 8) )
  {
    if ( !v13 )
      return v11;
    return (unsigned int)-1;
  }
  if ( !v13 )
    return 1;
  if ( v12 == 16 )
  {
    v56 = v11;
    v58 = *(_BYTE *)(v7 + 8);
    v39 = sub_1643030(*(_QWORD *)(v6 + 24));
    v10 = v58;
    v9 = 16;
    v11 = v56;
    v8 = (unsigned int)(*(_DWORD *)(v6 + 32) * v39);
    v40 = 0;
    if ( v58 != 16 )
      goto LABEL_25;
  }
  else
  {
    LODWORD(v8) = 0;
    if ( (_BYTE)v10 != 16 )
    {
LABEL_29:
      if ( (_BYTE)v9 == 15 )
      {
        if ( (_BYTE)v10 == 15 )
        {
          v11 = sub_1ACA9E0((__int64)a1, *(_DWORD *)(v6 + 8) >> 8, *(_DWORD *)(v7 + 8) >> 8);
          if ( v11 )
            return v11;
        }
        return 1;
      }
      if ( (_BYTE)v10 != 15 )
        return v11;
      return (unsigned int)-1;
    }
  }
  v55 = v11;
  v57 = v10;
  v59 = v9;
  v40 = *(_DWORD *)(v7 + 32) * sub_1643030(*(_QWORD *)(v7 + 24));
  v11 = v55;
  v10 = v57;
  v9 = v59;
  v8 = (unsigned int)v8;
LABEL_25:
  if ( v40 != (_DWORD)v8 )
    return sub_1ACA9E0((__int64)a1, v8, v40);
  if ( !v40 )
    goto LABEL_29;
LABEL_9:
  v60 = v11;
  v15 = sub_1593BB0(a2, v8, v9, v10);
  v11 = v60;
  if ( v15 )
  {
    v18 = sub_1593BB0(a3, v8, v16, v17);
    v11 = v60;
    if ( v18 )
      return v11;
  }
  v61 = v11;
  v19 = sub_1593BB0(a2, v8, v16, v17);
  v22 = v61;
  if ( v19 )
  {
    v23 = sub_1593BB0(a3, v8, v20, v21);
    v22 = v61;
    if ( !v23 )
      return 1;
  }
  v62 = v22;
  v24 = sub_1593BB0(a2, v8, v20, v21);
  v27 = v62;
  if ( !v24 )
  {
    v38 = sub_1593BB0(a3, v8, v25, v26);
    v27 = v62;
    if ( v38 )
      return (unsigned int)-1;
  }
  v28 = *(unsigned __int8 *)(a2 + 16);
  v29 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v28 <= 3u && (unsigned __int8)v29 <= 3u )
    return sub_1ACC5E0((__int64)a1, a2, a3);
  v63 = v27;
  v30 = sub_1ACA9E0((__int64)a1, v28, v29);
  if ( v30 )
    return v30;
  v11 = v63;
  v32 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned int)(v32 - 11) > 1 )
  {
    switch ( (char)v32 )
    {
      case 4:
        v11 = sub_1ACCBA0(a1, *(_QWORD *)(a2 - 48), *(_QWORD *)(a3 - 48), v31, v63);
        if ( v11 )
          return v11;
        v50 = *(_QWORD *)(a2 - 48);
        v51 = *(_QWORD *)(a2 - 24);
        v52 = *(_QWORD *)(a3 - 24);
        if ( *(_QWORD *)(a3 - 48) != v50 )
          return sub_1ACCBA0(a1, v51, *(_QWORD *)(a3 - 24), v49, v52);
        if ( v51 == v52 )
          return 0;
        v53 = *(_QWORD *)(v50 + 80);
        while ( 2 )
        {
          v54 = v53 - 24;
          if ( !v53 )
            v54 = 0;
          if ( v54 == v51 )
            return (unsigned int)-1;
          if ( v54 != v52 )
          {
            v53 = *(_QWORD *)(v53 + 8);
            continue;
          }
          break;
        }
        return 1;
      case 5:
        v43 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v11 = sub_1ACA9E0((__int64)a1, v43, *(_DWORD *)(a3 + 20) & 0xFFFFFFF);
        if ( v11 )
          return v11;
        if ( !v43 )
          return 0;
        v44 = 0;
        while ( 1 )
        {
          v11 = sub_1ACC630(
                  a1,
                  *(_QWORD *)(a2 + 24 * (v44 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                  *(_QWORD *)(a3 + 24 * (v44 - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))));
          if ( v11 )
            break;
          if ( v43 <= (unsigned int)++v44 )
            return 0;
        }
        return v11;
      case 6:
        v41 = *(_QWORD *)(v6 + 32);
        v11 = sub_1ACA9E0((__int64)a1, v41, *(_QWORD *)(v7 + 32));
        if ( v11 )
          return v11;
        v42 = 0;
        if ( !v41 )
          return 0;
        while ( 1 )
        {
          v11 = sub_1ACC630(
                  a1,
                  *(_QWORD *)(a2 + 24 * ((unsigned int)v42 - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                  *(_QWORD *)(a3 + 24 * ((unsigned int)v42 - (unsigned __int64)(*(_DWORD *)(a3 + 20) & 0xFFFFFFF))));
          if ( v11 )
            break;
          if ( v41 == ++v42 )
            return 0;
        }
        return v11;
      case 7:
        v47 = *(unsigned int *)(v6 + 12);
        v11 = sub_1ACA9E0((__int64)a1, v47, *(unsigned int *)(v7 + 12));
        if ( v11 )
          return v11;
        if ( !(_DWORD)v47 )
          return 0;
        v48 = 0;
        while ( 1 )
        {
          v11 = sub_1ACC630(
                  a1,
                  *(_QWORD *)(a2 + 24 * (v48 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                  *(_QWORD *)(a3 + 24 * (v48 - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))));
          if ( v11 )
            break;
          if ( ++v48 == v47 )
            return 0;
        }
        return v11;
      case 8:
        v45 = *(unsigned int *)(v6 + 32);
        v11 = sub_1ACA9E0((__int64)a1, v45, *(unsigned int *)(v7 + 32));
        if ( v11 )
          return v11;
        if ( !v45 )
          return 0;
        v46 = 0;
        break;
      case 9:
      case 10:
      case 11:
      case 12:
      case 15:
      case 16:
        return v11;
      case 13:
        return sub_1ACAA10((__int64)a1, a2 + 24, a3 + 24);
      case 14:
        return sub_1ACAA60((__int64)a1, a2 + 24, a3 + 24);
    }
    while ( 1 )
    {
      v11 = sub_1ACC630(
              a1,
              *(_QWORD *)(a2 + 24 * (v46 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
              *(_QWORD *)(a3 + 24 * (v46 - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))));
      if ( v11 )
        break;
      if ( v45 == ++v46 )
        return 0;
    }
    return v11;
  }
  v33 = (const void *)sub_1595920(a3);
  v35 = v34;
  v36 = (const void *)sub_1595920(a2);
  return sub_1ACABE0((__int64)a1, v36, v37, v33, v35);
}
