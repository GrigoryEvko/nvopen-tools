// Function: sub_14F8D00
// Address: 0x14f8d00
//
__int64 *__fastcall sub_14F8D00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // esi
  int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  char v15; // al
  __int64 *v16; // rax
  __int64 v17; // rdx
  int v19; // ebx
  unsigned int v20; // ebx
  __int64 v21; // rax
  __int64 v22; // r11
  unsigned int v23; // esi
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r9
  unsigned int v27; // edi
  _QWORD *v28; // rax
  __int64 v29; // r8
  unsigned __int64 v30; // rdx
  unsigned __int64 *v31; // r10
  unsigned __int64 v32; // r8
  unsigned int v33; // r12d
  const char *v34; // rax
  unsigned int v35; // edx
  __int64 v36; // r11
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdi
  char v40; // cl
  __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  __int64 v43; // rdi
  char v44; // al
  int v45; // eax
  __int64 *v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // edi
  int v50; // edi
  int v51; // edi
  int v52; // edi
  int v53; // r9d
  unsigned int v54; // esi
  _QWORD *v55; // r8
  __int64 v56; // r10
  int v57; // edi
  int v58; // edi
  int v59; // r9d
  _QWORD *v60; // r8
  unsigned int i; // esi
  __int64 v62; // r10
  unsigned int v63; // esi
  unsigned int v64; // esi
  unsigned __int64 v65; // [rsp+0h] [rbp-390h]
  int v66; // [rsp+10h] [rbp-380h]
  unsigned __int64 v67; // [rsp+10h] [rbp-380h]
  _QWORD *v68; // [rsp+18h] [rbp-378h]
  __int64 v69; // [rsp+18h] [rbp-378h]
  unsigned __int64 v70; // [rsp+18h] [rbp-378h]
  unsigned __int64 v71; // [rsp+20h] [rbp-370h]
  __int64 v72; // [rsp+28h] [rbp-368h]
  __int64 v73; // [rsp+28h] [rbp-368h]
  unsigned int v74; // [rsp+30h] [rbp-360h]
  unsigned __int64 v75; // [rsp+48h] [rbp-348h] BYREF
  __int64 v76[2]; // [rsp+50h] [rbp-340h] BYREF
  __int64 *v77; // [rsp+60h] [rbp-330h] BYREF
  char v78; // [rsp+68h] [rbp-328h]
  __int16 v79; // [rsp+70h] [rbp-320h]
  __int64 v80[2]; // [rsp+80h] [rbp-310h] BYREF
  __int64 v81; // [rsp+90h] [rbp-300h] BYREF
  const char *v82; // [rsp+C0h] [rbp-2D0h] BYREF
  __int64 v83; // [rsp+C8h] [rbp-2C8h]
  _WORD v84[64]; // [rsp+D0h] [rbp-2C0h] BYREF
  char *v85; // [rsp+150h] [rbp-240h] BYREF
  __int64 v86; // [rsp+158h] [rbp-238h]
  char v87; // [rsp+160h] [rbp-230h] BYREF
  char v88; // [rsp+161h] [rbp-22Fh]

  v5 = a2 + 32;
  if ( !a3 )
  {
LABEL_4:
    v11 = *(_DWORD *)(a2 + 68);
    if ( (unsigned __int8)sub_15127D0(v5, 14, 0) )
    {
      v88 = 1;
      v85 = "Invalid record";
      v87 = 3;
      sub_14EE4B0(a1, a2 + 8, (__int64)&v85);
    }
    else
    {
      v85 = &v87;
      v86 = 0x4000000000LL;
      v12 = *(_QWORD *)(a2 + 440);
      v84[0] = 260;
      v82 = (const char *)(v12 + 240);
      sub_16E1010(v80);
      v82 = (const char *)v84;
      v83 = 0x8000000000LL;
      while ( 1 )
      {
        v13 = sub_14ED070(v5, 0);
        if ( (_DWORD)v13 == 1 )
          break;
        if ( (v13 & 0xFFFFFFFD) == 0 )
        {
          HIBYTE(v79) = 1;
          v34 = "Malformed block";
LABEL_30:
          v77 = (__int64 *)v34;
          LOBYTE(v79) = 3;
          sub_14EE4B0(a1, a2 + 8, (__int64)&v77);
          goto LABEL_31;
        }
        LODWORD(v86) = 0;
        v14 = sub_1510D70(v5, HIDWORD(v13), &v85, 0);
        switch ( v14 )
        {
          case 2:
            if ( (unsigned __int8)sub_14EA1E0((__int64)v85, (unsigned int)v86, 1, (__int64)&v82)
              || (v41 = *(_QWORD *)(a2 + 1368), v42 = *(unsigned int *)v85, (*(_QWORD *)(a2 + 1376) - v41) >> 3 <= v42)
              || (v43 = *(_QWORD *)(v41 + 8 * v42)) == 0 )
            {
              HIBYTE(v79) = 1;
              v34 = "Invalid record";
              goto LABEL_30;
            }
            v76[0] = (__int64)v82;
            v76[1] = (unsigned int)v83;
            v79 = 261;
            v77 = v76;
            sub_164B780(v43, &v77);
            LODWORD(v83) = 0;
            break;
          case 3:
            sub_14EE7C0((__int64)&v77, (_QWORD *)a2, (__int64 **)&v85, 2u, (__int64)v80);
            v44 = v78;
            v78 &= ~2u;
            v45 = v44 & 1;
            if ( v45 )
            {
              v46 = v77;
              v77 = 0;
              v75 = (unsigned __int64)v46 | 1;
              if ( ((unsigned __int64)v46 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
                *a1 = 0;
                sub_14ECA50(a1, &v75);
                sub_14ECA90((__int64 *)&v75);
                sub_14F4320(&v77, (__int64)&v75, v47);
                goto LABEL_31;
              }
              v48 = 0;
            }
            else
            {
              v75 = 1;
              v76[0] = 0;
              sub_14ECA90(v76);
              v48 = (__int64)v77;
              LOBYTE(v45) = 0;
            }
            if ( !*(_BYTE *)(v48 + 16) )
            {
              v76[0] = v48;
              v65 = 32 * (*((_QWORD *)v85 + 1) - 1LL);
              sub_14F87D0(a2 + 1488, v76)[1] = v65 + (unsigned int)(v11 + 8);
              if ( v65 > *(_QWORD *)(a2 + 456) )
                *(_QWORD *)(a2 + 456) = v65;
              if ( (v78 & 2) != 0 )
                sub_14F42B0(&v77, (__int64)v76, v65);
              LOBYTE(v45) = v78 & 1;
            }
            if ( (_BYTE)v45 && v77 )
              (*(void (__fastcall **)(__int64 *))(*v77 + 8))(v77);
            break;
          case 1:
            sub_14EE7C0((__int64)&v77, (_QWORD *)a2, (__int64 **)&v85, 1u, (__int64)v80);
            v15 = v78;
            v78 &= ~2u;
            if ( (v15 & 1) != 0 )
            {
              v16 = v77;
              v77 = 0;
              v76[0] = (unsigned __int64)v16 | 1;
              if ( ((unsigned __int64)v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
                *a1 = 0;
                sub_14ECA50(a1, v76);
                sub_14ECA90(v76);
                sub_14F4320(&v77, (__int64)v76, v17);
                goto LABEL_31;
              }
            }
            break;
        }
      }
      if ( a3 )
      {
        *(_DWORD *)(a2 + 64) = 0;
        *(_QWORD *)(a2 + 48) = (v71 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v71 & 0x3F) != 0 )
          sub_14ECAB0(v5, v71 & 0x3F);
      }
      *a1 = 1;
LABEL_31:
      if ( v82 != (const char *)v84 )
        _libc_free((unsigned __int64)v82);
      if ( (__int64 *)v80[0] != &v81 )
        j_j___libc_free_0(v80[0], v81 + 1);
      if ( v85 != &v87 )
        _libc_free((unsigned __int64)v85);
    }
    return a1;
  }
  v7 = 8LL * *(_QWORD *)(a2 + 48);
  v8 = *(unsigned int *)(a2 + 64);
  *(_DWORD *)(a2 + 64) = 0;
  v71 = v7 - v8;
  v9 = (4 * a3) & 0x1FFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 48) = v9;
  v10 = (32 * (_BYTE)a3) & 0x3F;
  if ( ((32 * (_BYTE)a3) & 0x3F) == 0 )
    goto LABEL_3;
  v30 = *(_QWORD *)(a2 + 40);
  if ( v9 >= v30 )
    goto LABEL_55;
  v31 = (unsigned __int64 *)(v9 + *(_QWORD *)(a2 + 32));
  if ( v30 < v9 + 8 )
  {
    v35 = v30 - v9;
    *(_QWORD *)(a2 + 56) = 0;
    v36 = v35;
    v33 = 8 * v35;
    v37 = v35 + v9;
    if ( v35 )
    {
      v38 = 0;
      v32 = 0;
      do
      {
        v39 = *((unsigned __int8 *)v31 + v38);
        v40 = 8 * v38++;
        v32 |= v39 << v40;
        *(_QWORD *)(a2 + 56) = v32;
      }
      while ( v36 != v38 );
      *(_QWORD *)(a2 + 48) = v37;
      *(_DWORD *)(a2 + 64) = v33;
      if ( v10 <= v33 )
        goto LABEL_28;
    }
    else
    {
      *(_QWORD *)(a2 + 48) = v37;
    }
LABEL_55:
    sub_16BD130("Unexpected end of file", 1);
  }
  v32 = *v31;
  *(_QWORD *)(a2 + 48) = v9 + 8;
  v33 = 64;
LABEL_28:
  *(_DWORD *)(a2 + 64) = v33 - v10;
  *(_QWORD *)(a2 + 56) = v32 >> v10;
LABEL_3:
  sub_14ECC00(v5, 0);
  if ( !*(_BYTE *)(a2 + 392) )
    goto LABEL_4;
  v19 = *(_DWORD *)(a2 + 68);
  if ( (unsigned __int8)sub_15127D0(v5, 14, 0) )
  {
    v88 = 1;
    v85 = "Invalid record";
    v87 = 3;
    sub_14EE4B0(v80, a2 + 8, (__int64)&v85);
    goto LABEL_44;
  }
  v20 = v19 + 8;
  v85 = &v87;
  v86 = 0x4000000000LL;
  while ( 1 )
  {
    v21 = sub_14ED070(v5, 0);
    if ( (_DWORD)v21 == 1 )
      break;
    if ( (v21 & 0xFFFFFFFD) == 0 )
    {
      v82 = "Malformed block";
      v84[0] = 259;
      sub_14EE4B0(v80, a2 + 8, (__int64)&v82);
      goto LABEL_42;
    }
    LODWORD(v86) = 0;
    if ( (unsigned int)sub_1510D70(v5, HIDWORD(v21), &v85, 0) == 3 )
    {
      v22 = v20;
      v23 = *(_DWORD *)(a2 + 1512);
      v24 = 32 * (*((_QWORD *)v85 + 1) - 1LL);
      v25 = *(_QWORD *)(*(_QWORD *)(a2 + 552) + 24LL * *(unsigned int *)v85 + 16);
      if ( !v23 )
      {
        ++*(_QWORD *)(a2 + 1488);
LABEL_90:
        v70 = v24;
        v73 = v25;
        sub_14F8610(a2 + 1488, 2 * v23);
        v57 = *(_DWORD *)(a2 + 1512);
        if ( !v57 )
        {
LABEL_116:
          ++*(_DWORD *)(a2 + 1504);
          BUG();
        }
        v58 = v57 - 1;
        v25 = v73;
        v59 = 1;
        v24 = v70;
        v22 = v20;
        v60 = 0;
        for ( i = v58 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4)); ; i = v58 & v63 )
        {
          v28 = (_QWORD *)(*(_QWORD *)(a2 + 1496) + 16LL * i);
          v62 = *v28;
          if ( v73 == *v28 )
          {
            v50 = *(_DWORD *)(a2 + 1504) + 1;
            goto LABEL_82;
          }
          if ( v62 == -8 )
            break;
          if ( v62 != -16 || v60 )
            v28 = v60;
          v63 = v59 + i;
          v60 = v28;
          ++v59;
        }
        v50 = *(_DWORD *)(a2 + 1504) + 1;
        if ( v60 )
          v28 = v60;
        goto LABEL_82;
      }
      v26 = *(_QWORD *)(a2 + 1496);
      v74 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
      v27 = (v23 - 1) & v74;
      v28 = (_QWORD *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v25 == *v28 )
        goto LABEL_23;
      v66 = 1;
      v68 = 0;
      while ( v29 != -8 )
      {
        if ( !v68 )
        {
          if ( v29 != -16 )
            v28 = 0;
          v68 = v28;
        }
        v27 = (v23 - 1) & (v66 + v27);
        v28 = (_QWORD *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( v25 == *v28 )
          goto LABEL_23;
        ++v66;
      }
      if ( v68 )
        v28 = v68;
      v49 = *(_DWORD *)(a2 + 1504);
      ++*(_QWORD *)(a2 + 1488);
      v50 = v49 + 1;
      if ( 4 * v50 >= 3 * v23 )
        goto LABEL_90;
      if ( v23 - *(_DWORD *)(a2 + 1508) - v50 <= v23 >> 3 )
      {
        v67 = v24;
        v69 = v25;
        sub_14F8610(a2 + 1488, v23);
        v51 = *(_DWORD *)(a2 + 1512);
        if ( !v51 )
          goto LABEL_116;
        v52 = v51 - 1;
        v53 = 1;
        v25 = v69;
        v24 = v67;
        v22 = v20;
        v72 = *(_QWORD *)(a2 + 1496);
        v54 = v52 & v74;
        v28 = 0;
        while ( 1 )
        {
          v55 = (_QWORD *)(v72 + 16LL * v54);
          v56 = *v55;
          if ( v69 == *v55 )
          {
            v50 = *(_DWORD *)(a2 + 1504) + 1;
            v28 = (_QWORD *)(v72 + 16LL * v54);
            goto LABEL_82;
          }
          if ( v56 == -8 )
            break;
          if ( v28 || v56 != -16 )
            v55 = v28;
          v64 = v53 + v54;
          v28 = v55;
          ++v53;
          v54 = v52 & v64;
        }
        v50 = *(_DWORD *)(a2 + 1504) + 1;
        if ( !v28 )
          v28 = (_QWORD *)(v72 + 16LL * v54);
      }
LABEL_82:
      *(_DWORD *)(a2 + 1504) = v50;
      if ( *v28 != -8 )
        --*(_DWORD *)(a2 + 1508);
      *v28 = v25;
      v28[1] = 0;
LABEL_23:
      v28[1] = v24 + v22;
      if ( v24 > *(_QWORD *)(a2 + 456) )
        *(_QWORD *)(a2 + 456) = v24;
    }
  }
  v80[0] = 1;
LABEL_42:
  if ( v85 != &v87 )
    _libc_free((unsigned __int64)v85);
LABEL_44:
  if ( (v80[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v80[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    *(_DWORD *)(a2 + 64) = 0;
    *(_QWORD *)(a2 + 48) = (v71 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v71 & 0x3F) != 0 )
      sub_14ECAB0(v5, v71 & 0x3F);
    *a1 = 1;
  }
  return a1;
}
