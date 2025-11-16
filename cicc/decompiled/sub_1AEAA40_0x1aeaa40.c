// Function: sub_1AEAA40
// Address: 0x1aeaa40
//
__int64 __fastcall sub_1AEAA40(__int64 a1)
{
  __int64 *v2; // rdi
  unsigned int v3; // r14d
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  int v8; // edx
  __int64 *v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // r12
  __int64 *v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rsi
  __int64 v24; // r8
  __int64 *v25; // rax
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // esi
  __int64 v32; // r14
  __int64 *v33; // r15
  __int64 v34; // rsi
  __int64 *v35; // r12
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // r15
  __int64 v40; // rsi
  __int64 *v41; // rax
  __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 *v46; // rdx
  __int64 v47; // rsi
  unsigned __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 *v50; // [rsp+0h] [rbp-C0h]
  __int64 *v51; // [rsp+8h] [rbp-B8h]
  __int64 *v52; // [rsp+8h] [rbp-B8h]
  __int64 *v53; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD *v54; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v55; // [rsp+28h] [rbp-98h] BYREF
  __int64 *v56; // [rsp+30h] [rbp-90h] BYREF
  __int64 v57; // [rsp+38h] [rbp-88h]
  _BYTE v58[16]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v59[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v60; // [rsp+70h] [rbp-50h] BYREF
  __int64 v61; // [rsp+78h] [rbp-48h]
  __int64 v62; // [rsp+80h] [rbp-40h]

  v57 = 0x100000000LL;
  v56 = (__int64 *)v58;
  sub_1AEA440((__int64)&v56, a1);
  if ( !(_DWORD)v57 )
    goto LABEL_2;
  v5 = sub_15F2050(a1);
  v6 = sub_1632FA0(v5);
  v7 = sub_16498A0(a1);
  v8 = *(unsigned __int8 *)(a1 + 16);
  v59[1] = a1;
  v9 = (__int64 *)v7;
  v53 = (__int64 *)v7;
  v59[0] = &v53;
  v59[2] = v7;
  v54 = v59;
  v55 = v59;
  if ( (unsigned int)(v8 - 60) > 0xC )
  {
    if ( (_BYTE)v8 == 56 )
    {
      v11 = sub_1632FA0(v5);
      v12 = **(_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v12 + 8) == 16 )
        v12 = **(_QWORD **)(v12 + 16);
      LODWORD(v61) = 8 * sub_15A95A0(v11, *(_DWORD *)(v12 + 8) >> 8);
      if ( (unsigned int)v61 <= 0x40 )
        v60 = 0;
      else
        sub_16A4EF0((__int64)&v60, 0, 0);
      v13 = sub_1632FA0(v5);
      if ( (unsigned __int8)sub_15FA310(a1, v13, (__int64)&v60) )
      {
        v14 = v56;
        v15 = &v56[(unsigned int)v57];
        if ( v56 != v15 )
        {
          do
          {
            v17 = *v14;
            if ( (unsigned int)v61 <= 0x40 )
              v16 = v60 << (64 - (unsigned __int8)v61) >> (64 - (unsigned __int8)v61);
            else
              v16 = *(_QWORD *)v60;
            ++v14;
            sub_1AE7E00((__int64 *)&v54, v17, v16);
          }
          while ( v15 != v14 );
        }
      }
      if ( (unsigned int)v61 > 0x40 && v60 )
        j_j___libc_free_0_0(v60);
LABEL_64:
      v2 = v56;
      v3 = 1;
      goto LABEL_3;
    }
    if ( (unsigned int)(v8 - 35) > 0x11 )
    {
      if ( (_BYTE)v8 != 54 )
      {
LABEL_2:
        v2 = v56;
        v3 = 0;
        goto LABEL_3;
      }
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v35 = *(__int64 **)(a1 - 8);
      else
        v35 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v36 = sub_1624210(*v35);
      v37 = sub_1628DA0(v9, (__int64)v36);
      v2 = v56;
      v38 = v37;
      v50 = &v56[(unsigned int)v57];
      if ( v50 != v56 )
      {
        v52 = v56;
        do
        {
          v39 = *v52;
          v40 = sub_15C48E0(
                  *(_QWORD **)(*(_QWORD *)(*v52 + 24 * (2LL - (*(_DWORD *)(*v52 + 20) & 0xFFFFFFF))) + 24LL),
                  1,
                  0,
                  0,
                  0);
          v41 = (__int64 *)(v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF));
          if ( *v41 )
          {
            v42 = v41[1];
            v43 = v41[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v43 = v42;
            if ( v42 )
              *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
          }
          *v41 = v38;
          if ( v38 )
          {
            v44 = *(_QWORD *)(v38 + 8);
            v41[1] = v44;
            if ( v44 )
              *(_QWORD *)(v44 + 16) = (unsigned __int64)(v41 + 1) | *(_QWORD *)(v44 + 16) & 3LL;
            v41[2] = (v38 + 8) | v41[2] & 3;
            *(_QWORD *)(v38 + 8) = v41;
          }
          v45 = sub_1628DA0(v9, v40);
          v46 = (__int64 *)(v39 + 24 * (2LL - (*(_DWORD *)(v39 + 20) & 0xFFFFFFF)));
          if ( *v46 )
          {
            v47 = v46[1];
            v48 = v46[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v48 = v47;
            if ( v47 )
              *(_QWORD *)(v47 + 16) = *(_QWORD *)(v47 + 16) & 3LL | v48;
          }
          *v46 = v45;
          if ( v45 )
          {
            v49 = *(_QWORD *)(v45 + 8);
            v46[1] = v49;
            if ( v49 )
              *(_QWORD *)(v49 + 16) = (unsigned __int64)(v46 + 1) | *(_QWORD *)(v49 + 16) & 3LL;
            v46[2] = (v45 + 8) | v46[2] & 3;
            *(_QWORD *)(v45 + 8) = v46;
          }
          ++v52;
        }
        while ( v50 != v52 );
        goto LABEL_64;
      }
    }
    else
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v29 = *(_QWORD *)(a1 - 8);
      else
        v29 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v30 = *(_QWORD *)(v29 + 24);
      v3 = 0;
      v2 = v56;
      if ( *(_BYTE *)(v30 + 16) != 13 )
        goto LABEL_3;
      v31 = *(_DWORD *)(v30 + 32);
      if ( v31 > 0x40 )
        goto LABEL_3;
      v51 = &v56[(unsigned int)v57];
      v32 = (__int64)(*(_QWORD *)(v30 + 24) << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31);
      if ( v51 != v56 )
      {
        v33 = v56;
        while ( 1 )
        {
          v34 = *v33;
          switch ( v8 )
          {
            case '#':
              sub_1AE7E00((__int64 *)&v54, v34, v32);
              break;
            case '%':
              sub_1AE7E00((__int64 *)&v54, v34, -v32);
              break;
            case '\'':
              v61 = v32;
              v60 = 16;
              v62 = 30;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '*':
              v61 = v32;
              v60 = 16;
              v62 = 27;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '-':
              v61 = v32;
              v60 = 16;
              v62 = 29;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '/':
              v61 = v32;
              v60 = 16;
              v62 = 36;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '0':
              v61 = v32;
              v60 = 16;
              v62 = 37;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '1':
              v61 = v32;
              v60 = 16;
              v62 = 38;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '2':
              v61 = v32;
              v60 = 16;
              v62 = 26;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '3':
              v61 = v32;
              v60 = 16;
              v62 = 33;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            case '4':
              v61 = v32;
              v60 = 16;
              v62 = 39;
              sub_1AE8000((__int64 *)&v55, v34, &v60, 3);
              break;
            default:
              goto LABEL_2;
          }
          if ( v51 == ++v33 )
            goto LABEL_64;
          v8 = *(unsigned __int8 *)(a1 + 16);
        }
      }
    }
    v3 = 1;
    goto LABEL_3;
  }
  LOBYTE(v10) = sub_15FB940(a1, v6);
  v3 = v10;
  if ( (_BYTE)v10 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v18 = *(__int64 **)(a1 - 8);
    else
      v18 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v19 = v53;
    v20 = sub_1624210(*v18);
    v21 = sub_1628DA0(v19, (__int64)v20);
    v2 = v56;
    v22 = v21;
    v23 = &v56[(unsigned int)v57];
    if ( v23 == v56 )
      goto LABEL_3;
    v24 = v21 + 8;
    do
    {
      v25 = (__int64 *)(*v2 - 24LL * (*(_DWORD *)(*v2 + 20) & 0xFFFFFFF));
      if ( *v25 )
      {
        v26 = v25[1];
        v27 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v27 = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
      }
      *v25 = v22;
      if ( v22 )
      {
        v28 = *(_QWORD *)(v22 + 8);
        v25[1] = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v28 + 16) & 3LL;
        v25[2] = v24 | v25[2] & 3;
        *(_QWORD *)(v22 + 8) = v25;
      }
      ++v2;
    }
    while ( v2 != v23 );
  }
  v2 = v56;
LABEL_3:
  if ( v2 != (__int64 *)v58 )
    _libc_free((unsigned __int64)v2);
  return v3;
}
