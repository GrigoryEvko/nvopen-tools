// Function: sub_2D7E760
// Address: 0x2d7e760
//
__int64 __fastcall sub_2D7E760(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v6; // rax
  unsigned __int64 i; // rcx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // ebx
  __int64 v13; // rax
  bool v14; // zf
  _BYTE *v15; // rdx
  _QWORD *v16; // rax
  char v17; // dl
  _BYTE *v18; // r13
  _BYTE *v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rax
  unsigned __int8 *v22; // r12
  _QWORD *v23; // r15
  __int64 v24; // rax
  bool v25; // r15
  unsigned __int8 *v26; // rdx
  bool v27; // al
  unsigned int v28; // esi
  int v29; // eax
  unsigned __int64 v30; // r15
  int v31; // eax
  unsigned __int8 *v32; // rdx
  unsigned __int64 *v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rax
  _QWORD *v36; // r8
  unsigned __int8 v38; // [rsp+Fh] [rbp-121h]
  __int64 v39; // [rsp+18h] [rbp-118h]
  __int64 v40; // [rsp+28h] [rbp-108h]
  bool v41; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v42; // [rsp+30h] [rbp-100h]
  _BYTE *v43; // [rsp+48h] [rbp-E8h]
  __int64 v44; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v45; // [rsp+60h] [rbp-D0h]
  unsigned __int64 v46; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v47; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v48; // [rsp+78h] [rbp-B8h] BYREF
  unsigned __int64 v49[2]; // [rsp+80h] [rbp-B0h] BYREF
  unsigned __int8 *v50; // [rsp+90h] [rbp-A0h]
  unsigned __int64 v51; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-88h] BYREF
  unsigned __int8 *v53; // [rsp+B0h] [rbp-80h]
  unsigned __int8 *v54; // [rsp+B8h] [rbp-78h]
  __int64 v55; // [rsp+C0h] [rbp-70h]
  __int64 v56; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v57; // [rsp+D8h] [rbp-58h]
  _QWORD *v58; // [rsp+E0h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+E8h] [rbp-48h]
  _BYTE v60[48]; // [rsp+100h] [rbp-30h] BYREF

  if ( !a2 )
    return 0;
  v3 = sub_B14240(a2);
  v39 = v2;
  if ( v3 == v2 )
    return 0;
  while ( *(_BYTE *)(v3 + 32) )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( v3 == v2 )
      return 0;
  }
  v38 = 0;
  if ( v3 == v2 )
    return v38;
  v4 = v3;
  do
  {
    if ( (unsigned __int8)(*(_BYTE *)(v4 + 64) - 1) > 1u )
      goto LABEL_9;
    sub_B129C0(&v51, v4);
    v46 = v52;
    sub_B129C0(v49, v4);
    v6 = v49[0];
    v45 = v49[0];
    v48 = v49[0];
    if ( v46 == v49[0] )
    {
LABEL_81:
      v56 = 0;
LABEL_82:
      LOBYTE(v57) = v57 | 1;
      v57 &= 1u;
      if ( !v57 )
        goto LABEL_83;
LABEL_26:
      v15 = v60;
      v16 = &v58;
      do
      {
LABEL_27:
        if ( v16 )
          *v16 = -4096;
        ++v16;
      }
      while ( v16 != (_QWORD *)v15 );
      goto LABEL_30;
    }
    for ( i = 0; ; i = v8 )
    {
      v9 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v6 & 4) != 0 || !v9 )
        break;
      v8 = i + 1;
      v6 = v9 + 144;
      if ( v9 + 144 == v46 )
        goto LABEL_22;
LABEL_19:
      ;
    }
    v8 = i + 1;
    v6 = (v9 + 8) | 4;
    if ( v6 != v46 )
      goto LABEL_19;
LABEL_22:
    if ( !i )
      goto LABEL_81;
    _BitScanReverse64(&v10, i);
    v56 = 0;
    v11 = 1LL << (64 - ((unsigned __int8)v10 ^ 0x3Fu));
    if ( (unsigned int)v11 <= 4 )
      goto LABEL_82;
    _BitScanReverse((unsigned int *)&v11, v11 - 1);
    v12 = 1 << (32 - (v11 ^ 0x1F));
    if ( v12 <= 4 )
      goto LABEL_82;
    LOBYTE(v57) = v57 & 0xFE;
    v13 = sub_C7D670(8LL * v12, 8);
    v14 = (v57 & 1) == 0;
    v57 &= 1u;
    v58 = (_QWORD *)v13;
    v59 = v12;
    if ( !v14 )
      goto LABEL_26;
LABEL_83:
    v16 = v58;
    v15 = &v58[v59];
    if ( v58 != (_QWORD *)v15 )
      goto LABEL_27;
LABEL_30:
    v48 = v46;
    v47 = v45;
    sub_2D7E430((__int64)&v56, (__int64 *)&v47, &v48);
    if ( !((unsigned int)v57 >> 1) )
    {
      v17 = v57 & 1;
      if ( (v57 & 1) != 0 )
      {
        v35 = 4;
        v36 = &v58;
      }
      else
      {
        v36 = v58;
        v35 = v59;
      }
      v19 = &v36[v35];
      v18 = &v36[v35];
      goto LABEL_36;
    }
    v17 = v57 & 1;
    if ( (v57 & 1) != 0 )
    {
      v18 = v60;
      v19 = &v58;
      do
      {
LABEL_33:
        if ( *(_QWORD *)v19 != -8192 && *(_QWORD *)v19 != -4096 )
          break;
        v19 += 8;
      }
      while ( v19 != v18 );
LABEL_36:
      if ( !v17 )
      {
        v20 = (__int64)v58;
        v21 = v59;
        goto LABEL_38;
      }
      v43 = v60;
      if ( v19 == v60 )
        goto LABEL_9;
    }
    else
    {
      v20 = (__int64)v58;
      v21 = v59;
      v18 = &v58[v59];
      v19 = v58;
      if ( v58 != (_QWORD *)v18 )
        goto LABEL_33;
LABEL_38:
      v43 = (_BYTE *)(v20 + 8 * v21);
      if ( v19 == v43 )
        goto LABEL_64;
    }
    v41 = 0;
    v40 = v4;
    v44 = a1 + 104;
    while ( 2 )
    {
      v22 = *(unsigned __int8 **)v19;
      v53 = 0;
      v52 = 2;
      v54 = v22;
      if ( v22 + 4096 != 0 && v22 != 0 && v22 != (unsigned __int8 *)-8192LL )
        sub_BD73F0((__int64)&v52);
      v51 = (unsigned __int64)&unk_4A26638;
      v55 = a1 + 104;
      if ( !(unsigned __int8)sub_2D69820(v44, (__int64)&v51, &v48) )
      {
        v28 = *(_DWORD *)(a1 + 128);
        v29 = *(_DWORD *)(a1 + 120);
        v30 = v48;
        ++*(_QWORD *)(a1 + 104);
        v31 = v29 + 1;
        v49[0] = v30;
        if ( 4 * v31 >= 3 * v28 )
        {
          v28 *= 2;
        }
        else if ( v28 - *(_DWORD *)(a1 + 124) - v31 > v28 >> 3 )
        {
          goto LABEL_67;
        }
        sub_2D72DF0(v44, v28);
        sub_2D69820(v44, (__int64)&v51, v49);
        v30 = v49[0];
        v31 = *(_DWORD *)(a1 + 120) + 1;
LABEL_67:
        *(_DWORD *)(a1 + 120) = v31;
        if ( *(_QWORD *)(v30 + 24) == -4096 )
        {
          v32 = v54;
          v24 = -4096;
          v33 = (unsigned __int64 *)(v30 + 8);
          if ( v54 != (unsigned __int8 *)-4096LL )
            goto LABEL_72;
        }
        else
        {
          --*(_DWORD *)(a1 + 124);
          v32 = v54;
          v24 = *(_QWORD *)(v30 + 24);
          if ( v54 != (unsigned __int8 *)v24 )
          {
            v33 = (unsigned __int64 *)(v30 + 8);
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
            {
              sub_BD60C0(v33);
              v32 = v54;
            }
LABEL_72:
            *(_QWORD *)(v30 + 24) = v32;
            if ( v32 != 0 && v32 + 4096 != 0 && v32 != (unsigned __int8 *)-8192LL )
              sub_BD6050(v33, v52 & 0xFFFFFFFFFFFFFFF8LL);
            v24 = (__int64)v54;
          }
        }
        v34 = v55;
        v23 = (_QWORD *)(v30 + 40);
        *v23 = 6;
        v23[1] = 0;
        *(v23 - 1) = v34;
        v23[2] = 0;
        goto LABEL_45;
      }
      v23 = (_QWORD *)(v48 + 40);
      v24 = (__int64)v54;
LABEL_45:
      v51 = (unsigned __int64)&unk_49DB368;
      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
        sub_BD60C0(&v52);
      v49[0] = 6;
      v49[1] = 0;
      v50 = (unsigned __int8 *)v23[2];
      if ( v50 != 0 && v50 + 4096 != 0 && v50 != (unsigned __int8 *)-8192LL )
      {
        sub_BD6050(v49, *v23 & 0xFFFFFFFFFFFFFFF8LL);
        v25 = v50 + 0x2000 != 0 && v50 != 0 && v50 + 4096 != 0;
        if ( v25 )
        {
          v53 = v50;
          v51 = 6;
          v52 = 0;
          sub_BD6050(&v51, v49[0] & 0xFFFFFFFFFFFFFFF8LL);
          v26 = v53;
          if ( v53 )
          {
            if ( v53 != (unsigned __int8 *)-8192LL && v53 != (unsigned __int8 *)-4096LL )
            {
              v42 = v53;
              sub_BD60C0(&v51);
              v26 = v42;
            }
            sub_B13360(v40, v22, v26, 0);
            v41 = v25;
            v27 = v50 + 0x2000 != 0 && v50 + 4096 != 0 && v50 != 0;
          }
          else
          {
            v27 = v50 + 4096 != 0 && v50 != 0 && v50 + 0x2000 != 0;
          }
          if ( v27 )
            sub_BD60C0(v49);
        }
      }
      do
        v19 += 8;
      while ( v19 != v18 && (*(_QWORD *)v19 == -4096 || *(_QWORD *)v19 == -8192) );
      if ( v19 != v43 )
        continue;
      break;
    }
    v4 = v40;
    v38 |= v41;
    if ( (v57 & 1) == 0 )
    {
      v20 = (__int64)v58;
      v21 = v59;
LABEL_64:
      sub_C7D6A0(v20, 8 * v21, 8);
    }
LABEL_9:
    v4 = *(_QWORD *)(v4 + 8);
    if ( v4 == v39 )
      break;
    while ( *(_BYTE *)(v4 + 32) )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( v4 == v39 )
        return v38;
    }
  }
  while ( v4 != v39 );
  return v38;
}
