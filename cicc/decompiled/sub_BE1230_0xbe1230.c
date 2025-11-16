// Function: sub_BE1230
// Address: 0xbe1230
//
void __fastcall sub_BE1230(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  char v5; // al
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // r9
  _BYTE *v19; // r15
  char v20; // al
  unsigned __int8 *v21; // rax
  unsigned __int64 v22; // r9
  char v23; // al
  unsigned __int8 *v24; // rdx
  char v25; // al
  bool v26; // al
  __int64 v27; // rax
  unsigned __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int *v35; // rax
  char v36; // dl
  unsigned __int8 v37; // al
  const char *v38; // rax
  _BYTE **v39; // rsi
  __int64 *v40; // rax
  __int64 v41; // r12
  _BYTE *v42; // rax
  bool v43; // zf
  __int64 v44; // [rsp+8h] [rbp-F8h]
  __int64 v45; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v46; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v47; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v48; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v49; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v50; // [rsp+20h] [rbp-E0h]
  __int64 v51; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v52; // [rsp+20h] [rbp-E0h]
  _BYTE *v53; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v54; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v55[4]; // [rsp+40h] [rbp-C0h] BYREF
  char v56; // [rsp+60h] [rbp-A0h]
  char v57; // [rsp+61h] [rbp-9Fh]
  const char *v58; // [rsp+70h] [rbp-90h] BYREF
  char *v59; // [rsp+78h] [rbp-88h]
  __int64 v60; // [rsp+80h] [rbp-80h]
  int v61; // [rsp+88h] [rbp-78h]
  char v62; // [rsp+8Ch] [rbp-74h]
  char v63; // [rsp+90h] [rbp-70h] BYREF
  char v64; // [rsp+91h] [rbp-6Fh]

  v2 = (_BYTE *)a2;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(*(_QWORD *)(v3 + 72) + 80LL);
  if ( v4 && v3 == v4 - 24 )
  {
    v55[0] = (_BYTE *)a2;
    v64 = 1;
    v58 = "EH pad cannot be in entry block.";
    v63 = 3;
    sub_BE0C10(a1, (__int64)&v58, v55);
    return;
  }
  v5 = *(_BYTE *)a2;
  v6 = *(_QWORD *)(v3 + 16);
  if ( *(_BYTE *)a2 != 95 )
  {
    if ( v5 == 81 )
    {
      if ( v6 )
      {
        while ( (unsigned __int8)(**(_BYTE **)(v6 + 24) - 30) > 0xAu )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            goto LABEL_33;
        }
        v13 = sub_AA5510(*(_QWORD *)(a2 + 40));
        v14 = *(_QWORD *)(a2 - 32);
        if ( *(_QWORD *)(v14 + 40) != v13 )
        {
          v64 = 1;
          v58 = "Block containg CatchPadInst must be jumped to only by its catchswitch.";
          v63 = 3;
          sub_BDBF70(a1, (__int64)&v58);
          if ( !*a1 )
            return;
          goto LABEL_22;
        }
      }
      else
      {
LABEL_33:
        v14 = *(_QWORD *)(a2 - 32);
      }
      if ( (*(_BYTE *)(v14 + 2) & 1) == 0 )
        return;
      v15 = *(_QWORD *)(*(_QWORD *)(v14 - 8) + 32LL);
      if ( v15 != v3 )
        return;
      if ( !v15 )
        return;
      v64 = 1;
      v58 = "Catchswitch cannot unwind to one of its catchpads";
      v63 = 3;
      sub_BDBF70(a1, (__int64)&v58);
      if ( !*a1 )
        return;
      sub_BDBD80((__int64)a1, (_BYTE *)v14);
LABEL_22:
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
      return;
    }
    if ( (unsigned __int8)(v5 - 80) <= 1u )
      v44 = *(_QWORD *)(a2 - 32);
    else
      v44 = **(_QWORD **)(a2 - 8);
    if ( !v6 )
      return;
    while ( 1 )
    {
      v16 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v16 - 30) <= 0xAu )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return;
    }
LABEL_40:
    v17 = *(_QWORD *)(v16 + 40);
    v18 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v18 == v17 + 48 )
      goto LABEL_89;
    if ( !v18 )
LABEL_111:
      BUG();
    v19 = (_BYTE *)(v18 - 24);
    if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
    {
LABEL_89:
      v53 = 0;
      BUG();
    }
    v53 = (_BYTE *)(v18 - 24);
    v20 = *(_BYTE *)(v18 - 24);
    switch ( v20 )
    {
      case '"':
        if ( v3 != *(_QWORD *)(v18 - 88) || v3 == *(_QWORD *)(v18 - 120) )
        {
          v64 = 1;
          v58 = "EH pad must be jumped to via an unwind edge";
          v63 = 3;
          sub_BDBF70(a1, (__int64)&v58);
          if ( *a1 )
          {
            sub_BDBD80((__int64)a1, v2);
            sub_BDBD80((__int64)a1, v19);
          }
          return;
        }
        v48 = v18;
        v21 = sub_BD3990(*(unsigned __int8 **)(v18 - 56), a2);
        v22 = v48;
        if ( !*v21 )
        {
          v49 = v21;
          if ( (v21[33] & 0x20) != 0 )
          {
            a2 = 41;
            v46 = v22;
            v23 = sub_A73ED0((_QWORD *)(v22 + 48), 41);
            v22 = v46;
            v24 = v49;
            if ( v23 || (a2 = 41, v25 = sub_B49560((__int64)v19, 41), v24 = v49, v22 = v46, v25) )
            {
              v50 = v22;
              v26 = sub_B58D90(*((_DWORD *)v24 + 9));
              v22 = v50;
              if ( !v26 )
                goto LABEL_51;
            }
          }
        }
        if ( *(char *)(v22 - 17) >= 0 )
          goto LABEL_88;
        v47 = v22;
        v27 = sub_BD2BC0((__int64)v19);
        v28 = v47;
        v30 = v29 + v27;
        v51 = v30;
        if ( *(char *)(v47 - 17) >= 0 )
        {
          v32 = v30 >> 4;
        }
        else
        {
          v31 = sub_BD2BC0((__int64)v19);
          v28 = v47;
          v32 = (v51 - v31) >> 4;
        }
        if ( (_DWORD)v32 )
        {
          v33 = 0;
          v45 = 16LL * (unsigned int)v32;
          while ( 1 )
          {
            v34 = 0;
            if ( *(char *)(v28 - 17) < 0 )
            {
              v52 = v28;
              v34 = sub_BD2BC0((__int64)v19);
              v28 = v52;
            }
            v35 = (unsigned int *)(v33 + v34);
            if ( *(_DWORD *)(*(_QWORD *)v35 + 8LL) == 1 )
              break;
            v33 += 16;
            if ( v33 == v45 )
              goto LABEL_88;
          }
          a2 = *(_QWORD *)&v19[32 * (v35[2] - (unsigned __int64)(*(_DWORD *)(v28 - 20) & 0x7FFFFFF))];
          v54 = (unsigned __int8 *)a2;
        }
        else
        {
LABEL_88:
          v40 = (__int64 *)sub_BD5C60((__int64)v19);
          v54 = (unsigned __int8 *)sub_AC3540(v40);
          a2 = (__int64)v54;
        }
        break;
      case '%':
        a2 = *(_QWORD *)(v18 - 32LL * (*(_DWORD *)(v18 - 20) & 0x7FFFFFF) - 24);
        v54 = (unsigned __int8 *)a2;
        if ( a2 == v44 )
        {
          v41 = *a1;
          v64 = 1;
          v58 = "A cleanupret must exit its cleanup";
          v63 = 3;
          if ( v41 )
          {
            sub_CA0E80(&v58, v41);
            v42 = *(_BYTE **)(v41 + 32);
            if ( (unsigned __int64)v42 >= *(_QWORD *)(v41 + 24) )
            {
              sub_CB5D20(v41, 10);
            }
            else
            {
              *(_QWORD *)(v41 + 32) = v42 + 1;
              *v42 = 10;
            }
          }
          v43 = *a1 == 0;
          *((_BYTE *)a1 + 152) = 1;
          if ( !v43 )
            sub_BDBD80((__int64)a1, v19);
          return;
        }
        break;
      case '\'':
        v54 = (unsigned __int8 *)(v18 - 24);
        a2 = v18 - 24;
        break;
      default:
        v64 = 1;
        v58 = "EH pad must be jumped to via an unwind edge";
        v63 = 3;
        sub_BDBF70(a1, (__int64)&v58);
        if ( !*a1 )
          return;
        sub_BDBD80((__int64)a1, v2);
        a2 = (__int64)v53;
        if ( !v53 )
          return;
        goto LABEL_22;
    }
    v60 = 8;
    v58 = 0;
    v59 = &v63;
    v61 = 0;
    v62 = 1;
    if ( v2 == (_BYTE *)a2 )
    {
LABEL_83:
      v39 = v55;
      v57 = 1;
      v55[0] = "EH pad cannot handle exceptions raised within it";
      v56 = 3;
      sub_BDBF70(a1, (__int64)v55);
      if ( *a1 )
      {
        if ( v54 )
          sub_BDBD80((__int64)a1, v54);
        v39 = (_BYTE **)v53;
        if ( v53 )
          sub_BDBD80((__int64)a1, v53);
      }
LABEL_73:
      if ( !v62 )
        _libc_free(v59, v39);
    }
    else
    {
      while ( v44 != a2 )
      {
        if ( *(_BYTE *)a2 == 21 )
        {
          v57 = 1;
          v38 = "A single unwind edge may only enter one EH pad";
          goto LABEL_72;
        }
        sub_AE6EC0((__int64)&v58, a2);
        if ( !v36 )
        {
          v57 = 1;
          v39 = v55;
          v55[0] = "EH pad jumps through a cycle of pads";
          v56 = 3;
          sub_BE1130(a1, (__int64)v55, &v54);
          goto LABEL_73;
        }
        v37 = *v54;
        if ( *v54 <= 0x1Cu )
          goto LABEL_71;
        if ( (unsigned int)v37 - 80 <= 1 )
        {
          a2 = *((_QWORD *)v54 - 4);
        }
        else
        {
          if ( v37 != 39 )
          {
LABEL_71:
            v57 = 1;
            v38 = "Parent pad must be catchpad/cleanuppad/catchswitch";
LABEL_72:
            v39 = v55;
            v55[0] = v38;
            v56 = 3;
            sub_BE0C10(a1, (__int64)v55, &v53);
            goto LABEL_73;
          }
          a2 = **((_QWORD **)v54 - 1);
        }
        v54 = (unsigned __int8 *)a2;
        if ( v2 == (_BYTE *)a2 )
          goto LABEL_83;
      }
      if ( !v62 )
        _libc_free(v59, a2);
LABEL_51:
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
        v16 = *(_QWORD *)(v6 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v16 - 30) <= 0xAu )
          goto LABEL_40;
      }
    }
    return;
  }
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return;
    }
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 40);
      v9 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 == v8 + 48 || !v9 || (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
        goto LABEL_111;
      if ( *(_BYTE *)(v9 - 24) != 34 || *(_QWORD *)(v9 - 88) != v3 || *(_QWORD *)(v9 - 120) == v3 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return;
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
    v10 = *a1;
    v64 = 1;
    v58 = "Block containing LandingPadInst must be jumped to only by the unwind edge of an invoke.";
    v63 = 3;
    if ( v10 )
    {
      sub_CA0E80(&v58, v10);
      v11 = *(_BYTE **)(v10 + 32);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
      {
        sub_CB5D20(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 32) = v11 + 1;
        *v11 = 10;
      }
      v12 = *a1;
      *((_BYTE *)a1 + 152) = 1;
      if ( v12 )
        goto LABEL_22;
    }
    else
    {
      *((_BYTE *)a1 + 152) = 1;
    }
  }
}
