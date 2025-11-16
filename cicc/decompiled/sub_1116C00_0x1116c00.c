// Function: sub_1116C00
// Address: 0x1116c00
//
_QWORD *__fastcall sub_1116C00(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  char v4; // r15
  __int64 v5; // r12
  bool v6; // al
  unsigned __int64 v7; // rbx
  unsigned __int8 v8; // cl
  _QWORD *v9; // r13
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // r15d
  _QWORD *v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdx
  unsigned __int8 v19; // si
  bool v20; // r9
  unsigned __int8 v21; // al
  bool v22; // al
  bool v23; // r9
  bool v24; // r13
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // ebx
  unsigned int v29; // eax
  int v30; // esi
  int v31; // ebx
  _QWORD *v32; // rax
  bool v33; // al
  unsigned __int8 v34; // cl
  __int16 v35; // r14
  _QWORD *v36; // rax
  __int64 v37; // rdi
  bool v38; // al
  unsigned __int8 v39; // si
  bool v40; // al
  __int16 v41; // r14
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r14
  _QWORD *v45; // rax
  __int64 *v46; // rdi
  __int16 v47; // r14
  unsigned int **v48; // rdi
  __int16 v49; // r14
  __int64 v50; // r15
  __int64 v51; // r12
  _QWORD *v52; // rax
  __int64 **v53; // [rsp+0h] [rbp-B0h]
  bool v54; // [rsp+0h] [rbp-B0h]
  __int64 v55; // [rsp+0h] [rbp-B0h]
  __int64 v56; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v57; // [rsp+0h] [rbp-B0h]
  bool v58; // [rsp+0h] [rbp-B0h]
  unsigned int v59; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v60; // [rsp+8h] [rbp-A8h]
  bool v61; // [rsp+Fh] [rbp-A1h]
  char v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v64; // [rsp+10h] [rbp-A0h]
  bool v65; // [rsp+10h] [rbp-A0h]
  __int64 v66; // [rsp+10h] [rbp-A0h]
  int v68[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v69; // [rsp+40h] [rbp-70h]
  _BYTE v70[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v71; // [rsp+70h] [rbp-40h]

  v3 = *(unsigned __int8 **)(a2 - 64);
  v4 = *v3;
  if ( *v3 != 68 && v4 != 69 )
    return 0;
  v5 = *((_QWORD *)v3 - 4);
  if ( !v5 )
    return 0;
  v62 = v4 == 69;
  v6 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
  v7 = *(_QWORD *)(a2 - 32);
  v61 = v6;
  v8 = *(_BYTE *)v7;
  if ( *(_BYTE *)v7 <= 0x1Cu )
  {
    if ( v8 > 0x15u )
      return 0;
    v53 = *(__int64 ***)(*((_QWORD *)v3 - 4) + 8LL);
    v59 = *v3 - 29;
    v11 = sub_AD4C30(v7, v53, 0);
    v12 = sub_96F480(v59, v11, *(_QWORD *)(v7 + 8), *(_QWORD *)(a1 + 88));
    if ( v7 == v12 && v12 != 0 && v11 )
    {
      v13 = *(_WORD *)(a2 + 2) & 0x3F;
      if ( (unsigned int)(v13 - 32) <= 1 || v62 && v61 )
      {
        v71 = 257;
        v14 = sub_BD2C40(72, unk_3F10FD0);
        v9 = v14;
        if ( v14 )
          sub_1113300((__int64)v14, v13, v5, v11, (__int64)v70);
      }
      else
      {
        v35 = sub_B52EF0(v13);
        v71 = 257;
        v36 = sub_BD2C40(72, unk_3F10FD0);
        v9 = v36;
        if ( v36 )
          sub_1113300((__int64)v36, v35, v5, v11, (__int64)v70);
      }
    }
    else
    {
      if ( v4 != 69 || v61 || *(_BYTE *)v7 != 17 )
        return 0;
      if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x24 )
      {
        v44 = sub_AD62B0((__int64)v53);
        v71 = 257;
        v45 = sub_BD2C40(72, unk_3F10FD0);
        v9 = v45;
        if ( v45 )
          sub_1113300((__int64)v45, 38, v5, v44, (__int64)v70);
      }
      else
      {
        v15 = sub_AD6530((__int64)v53, v11);
        v71 = 257;
        v16 = sub_BD2C40(72, unk_3F10FD0);
        v9 = v16;
        if ( v16 )
          sub_1113300((__int64)v16, 40, v5, v15, (__int64)v70);
      }
    }
  }
  else
  {
    if ( v8 != 68 && v8 != 69 )
      return 0;
    v17 = *(_QWORD *)(v7 - 32);
    if ( !v17 )
      return 0;
    v18 = *(_QWORD *)(a2 - 64);
    v19 = *(_BYTE *)v18;
    v20 = *(_BYTE *)v18 == 68;
    if ( v20 == (v8 == 68) )
      goto LABEL_33;
    if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
    {
      if ( v19 > 0x1Cu && ((v19 - 68) & 0xFB) == 0 )
      {
        v21 = *(_BYTE *)v7;
        goto LABEL_30;
      }
      v21 = *(_BYTE *)v7;
      goto LABEL_46;
    }
    v37 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 <= 1 )
      v37 = **(_QWORD **)(v37 + 16);
    v57 = *(_BYTE *)v7;
    v65 = *(_BYTE *)v18 == 68;
    v38 = sub_BCAC40(v37, 1);
    v20 = v65;
    v8 = v57;
    if ( !v38 )
    {
      v7 = *(_QWORD *)(a2 - 32);
      v18 = *(_QWORD *)(a2 - 64);
      v21 = *(_BYTE *)v7;
      v39 = *(_BYTE *)v18;
      goto LABEL_57;
    }
    v43 = *(_QWORD *)(v17 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17 <= 1 )
      v43 = **(_QWORD **)(v43 + 16);
    if ( !sub_BCAC40(v43, 1) )
    {
      v7 = *(_QWORD *)(a2 - 32);
      v18 = *(_QWORD *)(a2 - 64);
      v20 = v65;
      v8 = v57;
      v21 = *(_BYTE *)v7;
      v39 = *(_BYTE *)v18;
LABEL_57:
      if ( v39 > 0x1Cu && ((v39 - 68) & 0xFB) == 0 )
      {
LABEL_30:
        if ( v21 <= 0x1Cu || ((v21 - 68) & 0xFB) != 0 )
        {
          v54 = v20;
          v63 = v18;
          v22 = sub_B44910(v18);
          v23 = v54;
          v18 = v63;
          v24 = v22;
        }
        else
        {
          v58 = v20;
          v66 = v18;
          v60 = v8;
          v24 = sub_B44910(v18);
          v40 = sub_B44910(v7);
          v18 = v66;
          v23 = v58;
          if ( v40 )
          {
            v34 = v60;
            v62 = v58 && v24;
            if ( !v58 || !v24 )
            {
LABEL_63:
              if ( v34 != 68 )
                return 0;
              v62 = 1;
            }
LABEL_33:
            v25 = *(_QWORD *)(v17 + 8);
            if ( v25 == *(_QWORD *)(v5 + 8) )
            {
LABEL_40:
              v31 = *(_WORD *)(a2 + 2) & 0x3F;
              if ( (unsigned int)(v31 - 32) <= 1 || v62 && v61 )
              {
                v71 = 257;
                v32 = sub_BD2C40(72, unk_3F10FD0);
                v9 = v32;
                if ( v32 )
                  sub_1113300((__int64)v32, v31, v5, v17, (__int64)v70);
              }
              else
              {
                v41 = sub_B52EF0(v31);
                v71 = 257;
                v42 = sub_BD2C40(72, unk_3F10FD0);
                v9 = v42;
                if ( v42 )
                  sub_1113300((__int64)v42, v41, v5, v17, (__int64)v70);
              }
              return v9;
            }
            v26 = *(_QWORD *)(v18 + 16);
            if ( v26 && !*(_QWORD *)(v26 + 8) || (v27 = *(_QWORD *)(v7 + 16)) != 0 && !*(_QWORD *)(v27 + 8) )
            {
              v55 = *(_QWORD *)(v5 + 8);
              v28 = sub_BCB060(v55);
              v29 = sub_BCB060(v25);
              v30 = (v62 & 1) + 39;
              if ( v28 < v29 )
              {
                v46 = *(__int64 **)(a1 + 32);
                v71 = 257;
                v5 = sub_1113450(v46, v30, v5, v25, (__int64)v70, 0, v68[0], 0);
                goto LABEL_40;
              }
              if ( v28 > v29 )
              {
                v71 = 257;
                v17 = sub_1113450(*(__int64 **)(a1 + 32), v30, v17, v55, (__int64)v70, 0, v68[0], 0);
                goto LABEL_40;
              }
            }
            return 0;
          }
        }
        v62 = v24 && v23;
        if ( !v24 || !v23 )
          return 0;
        goto LABEL_33;
      }
      if ( v21 <= 0x1Cu )
        return 0;
LABEL_46:
      if ( ((v21 - 68) & 0xFB) != 0 )
        return 0;
      v56 = v18;
      v64 = v8;
      v33 = sub_B44910(v7);
      v34 = v64;
      v18 = v56;
      if ( !v33 )
        return 0;
      goto LABEL_63;
    }
    v47 = *(_WORD *)(a2 + 2);
    v48 = *(unsigned int ***)(a1 + 32);
    v69 = 257;
    v49 = v47 & 0x3F;
    v50 = sub_A82480(v48, (_BYTE *)v5, (_BYTE *)v17, (__int64)v68);
    v51 = sub_AD6530(*(_QWORD *)(v5 + 8), v5);
    v71 = 257;
    v52 = sub_BD2C40(72, unk_3F10FD0);
    v9 = v52;
    if ( v52 )
      sub_1113300((__int64)v52, v49, v50, v51, (__int64)v70);
  }
  return v9;
}
