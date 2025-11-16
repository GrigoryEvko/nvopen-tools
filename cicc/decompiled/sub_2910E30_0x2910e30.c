// Function: sub_2910E30
// Address: 0x2910e30
//
__int64 __fastcall sub_2910E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  _QWORD *v6; // r14
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // r15
  char v11; // dl
  __int64 (__fastcall *v12)(_QWORD *); // rax
  unsigned __int64 v13; // rdi
  char v14; // dl
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r9
  _QWORD *v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r12
  __int64 v24; // r14
  int v25; // esi
  int v26; // esi
  const char *i; // r12
  __int64 *v28; // rbx
  _DWORD *v29; // r12
  __int64 v30; // r13
  __int64 v31; // rax
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // rdi
  _QWORD *v36; // r14
  __int64 v37; // r8
  __int64 v38; // r9
  _QWORD *v39; // r15
  _QWORD *j; // rbx
  _QWORD *k; // r12
  char v42; // al
  __int64 v43; // rbx
  __int64 v44; // rsi
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 *v50; // rax
  unsigned __int64 v51; // rbx
  unsigned __int64 v52; // r13
  unsigned __int64 v53; // rdi
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  int v58; // r14d
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rdx
  __int64 v63; // r10
  __int64 *v64; // rcx
  __int64 v65; // r14
  __int64 v66; // rdx
  _QWORD *v67; // r15
  __int64 v68; // r12
  __int64 v69; // rbx
  __int64 *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  _QWORD *v76; // [rsp+18h] [rbp-118h]
  _QWORD *v77; // [rsp+20h] [rbp-110h]
  char v78; // [rsp+28h] [rbp-108h]
  _QWORD *v79; // [rsp+28h] [rbp-108h]
  char v80; // [rsp+30h] [rbp-100h]
  char v81; // [rsp+30h] [rbp-100h]
  __int64 v82; // [rsp+30h] [rbp-100h]
  _QWORD *v83; // [rsp+30h] [rbp-100h]
  char v84; // [rsp+30h] [rbp-100h]
  __int64 v85; // [rsp+38h] [rbp-F8h]
  __int64 v86; // [rsp+38h] [rbp-F8h]
  __int64 *v87; // [rsp+38h] [rbp-F8h]
  __int64 v88; // [rsp+48h] [rbp-E8h] BYREF
  _QWORD v89[3]; // [rsp+50h] [rbp-E0h] BYREF
  int v90; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v91; // [rsp+70h] [rbp-C0h]
  __int64 v92; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v93; // [rsp+98h] [rbp-98h]
  int v94; // [rsp+A0h] [rbp-90h] BYREF
  int v95; // [rsp+A4h] [rbp-8Ch]
  int v96; // [rsp+A8h] [rbp-88h]
  char v97; // [rsp+ACh] [rbp-84h]
  _QWORD v98[2]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v99; // [rsp+C0h] [rbp-70h] BYREF
  _BYTE *v100; // [rsp+C8h] [rbp-68h]
  __int64 v101; // [rsp+D0h] [rbp-60h]
  int v102; // [rsp+D8h] [rbp-58h]
  char v103; // [rsp+DCh] [rbp-54h]
  _BYTE v104[80]; // [rsp+E0h] [rbp-50h] BYREF

  v4 = a3 + 24;
  v5 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  if ( *(_QWORD *)(v4 + 8) == v4 )
    goto LABEL_78;
  v85 = v5;
  v78 = 0;
  v6 = *(_QWORD **)(v4 + 8);
  do
  {
    v7 = (__int64)(v6 - 7);
    if ( !v6 )
      v7 = 0;
    v8 = v7;
    if ( !sub_B2FC80(v7) && v8 + 72 != (*(_QWORD *)(v8 + 72) & 0xFFFFFFFFFFFFFFF8LL) && (*(_BYTE *)(v8 + 3) & 0x40) != 0 )
    {
      v9 = sub_B2DBE0(v8);
      sub_E3FC80((__int64)&v92, *(_BYTE **)v9, *(_QWORD *)(v9 + 8));
      v10 = v92;
      v11 = *(_BYTE *)(v92 + 41);
      v12 = *(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)v92 + 8LL);
      if ( v12 == sub_BD9990 )
      {
        v13 = *(_QWORD *)(v92 + 8);
        *(_QWORD *)v92 = &unk_49DB390;
        if ( v13 != v10 + 24 )
        {
          v80 = v11;
          j_j___libc_free_0(v13);
          v11 = v80;
        }
        v81 = v11;
        j_j___libc_free_0(v10);
        v14 = v81;
      }
      else
      {
        v84 = *(_BYTE *)(v92 + 41);
        v12((_QWORD *)v92);
        v14 = v84;
      }
      if ( v14 )
      {
        v82 = sub_BC1CD0(v85, &unk_4F81450, v8);
        v15 = sub_BC1CD0(v85, &unk_4F89C30, v8);
        v16 = sub_BC1CD0(v85, &unk_4F6D3F8, v8);
        v78 |= sub_290FB60(a2, v8, v82 + 8, (__int64 *)(v15 + 8), (__int64 *)(v16 + 8), v17);
      }
    }
    v6 = (_QWORD *)v6[1];
  }
  while ( (_QWORD *)v4 != v6 );
  v76 = v6;
  if ( !v78 )
  {
LABEL_78:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v18 = *(_QWORD **)(a3 + 32);
  if ( v6 != v18 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v18 )
        {
          sub_B2BE50(0);
          BUG();
        }
        v19 = (__int64 *)sub_B2BE50((__int64)(v18 - 7));
        v20 = *((unsigned int *)v18 - 5);
        if ( !(_DWORD)v20 )
          break;
        v18[8] = sub_B612D0(v19, v20);
        v18 = (_QWORD *)v18[1];
        if ( v76 == v18 )
          goto LABEL_49;
      }
      sub_28FF340((__int64)&v92);
      if ( (*((_BYTE *)v18 - 54) & 1) != 0 )
      {
        sub_B2C6D0((__int64)(v18 - 7), v20, v21, v22);
        v23 = v18[5];
        v24 = v23 + 40LL * v18[6];
        if ( (*((_BYTE *)v18 - 54) & 1) != 0 )
        {
          sub_B2C6D0((__int64)(v18 - 7), v20, v48, v49);
          v23 = v18[5];
        }
      }
      else
      {
        v23 = v18[5];
        v24 = v23 + 40LL * v18[6];
      }
      while ( v23 != v24 )
      {
        while ( *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) != 14 )
        {
          v23 += 40;
          if ( v23 == v24 )
            goto LABEL_26;
        }
        v25 = *(_DWORD *)(v23 + 32);
        v23 += 40;
        sub_B2D5D0((__int64)(v18 - 7), v25, (__int64)&v92);
      }
LABEL_26:
      if ( *(_BYTE *)(**(_QWORD **)(*(v18 - 4) + 16LL) + 8LL) == 14 )
        sub_B2D550((__int64)(v18 - 7), (__int64)&v92);
      v26 = 92;
      for ( i = "\\"; ; v26 = *(_DWORD *)i )
      {
        i += 4;
        sub_B2D470((__int64)(v18 - 7), v26);
        if ( i == "<preserve-cfg>" )
          break;
      }
      v33 = v98[0];
      while ( v33 )
      {
        v34 = v33;
        sub_28FFC40(*(_QWORD **)(v33 + 24));
        v35 = *(_QWORD *)(v33 + 32);
        v33 = *(_QWORD *)(v33 + 16);
        if ( v35 != v34 + 56 )
          _libc_free(v35);
        j_j___libc_free_0(v34);
      }
      v18 = (_QWORD *)v18[1];
    }
    while ( v76 != v18 );
LABEL_49:
    v77 = *(_QWORD **)(a3 + 32);
    if ( v76 != v77 )
    {
      while ( 1 )
      {
        if ( !v77 )
          BUG();
        v36 = v77 + 2;
        v83 = v77 + 2;
        if ( v77 + 2 == (_QWORD *)(v77[2] & 0xFFFFFFFFFFFFFFF8LL) )
          goto LABEL_37;
        v88 = sub_B2BE50((__int64)(v77 - 7));
        v92 = (__int64)&v94;
        v93 = 0xC00000000LL;
        v39 = (_QWORD *)v77[3];
        if ( v36 == v39 )
        {
          j = 0;
        }
        else
        {
          if ( !v39 )
            BUG();
          for ( j = (_QWORD *)v39[4]; j == v39 + 3; j = (_QWORD *)v39[4] )
          {
            v39 = (_QWORD *)v39[1];
            if ( v36 == v39 )
              goto LABEL_32;
            if ( !v39 )
              BUG();
          }
        }
        if ( v83 != v39 )
          break;
LABEL_32:
        v28 = (__int64 *)v92;
        v29 = (_DWORD *)(v92 + 8LL * (unsigned int)v93);
        if ( (_DWORD *)v92 != v29 )
        {
          do
          {
            v30 = *v28++;
            v31 = sub_ACADE0(*(__int64 ***)(v30 + 8));
            sub_BD84D0(v30, v31);
            sub_B43D60((_QWORD *)v30);
          }
          while ( v29 != (_DWORD *)v28 );
          v29 = (_DWORD *)v92;
        }
        if ( v29 != &v94 )
          _libc_free((unsigned __int64)v29);
LABEL_37:
        v77 = (_QWORD *)v77[1];
        if ( v76 == v77 )
          goto LABEL_38;
      }
      k = j;
      while ( 1 )
      {
        if ( !k )
          BUG();
        v42 = *((_BYTE *)k - 24);
        v43 = (__int64)(k - 3);
        if ( v42 != 85 )
        {
          if ( (*((_BYTE *)k - 17) & 0x20) != 0 )
            goto LABEL_63;
          goto LABEL_66;
        }
        v71 = *(k - 7);
        if ( !v71
          || *(_BYTE *)v71
          || *(_QWORD *)(v71 + 24) != k[7]
          || (*(_BYTE *)(v71 + 33) & 0x20) == 0
          || *(_DWORD *)(v71 + 36) != 205 )
        {
          break;
        }
        v72 = (unsigned int)v93;
        v73 = (unsigned int)v93 + 1LL;
        if ( v73 > HIDWORD(v93) )
        {
          sub_C8D5F0((__int64)&v92, &v94, v73, 8u, v37, v38);
          v72 = (unsigned int)v93;
        }
        *(_QWORD *)(v92 + 8 * v72) = v43;
        LODWORD(v93) = v93 + 1;
LABEL_86:
        for ( k = (_QWORD *)k[1]; ; k = (_QWORD *)v39[4] )
        {
          v54 = v39 - 3;
          if ( !v39 )
            v54 = 0;
          if ( k != v54 + 6 )
            break;
          v39 = (_QWORD *)v39[1];
          if ( v83 == v39 )
            goto LABEL_32;
          if ( !v39 )
            BUG();
        }
        if ( v83 == v39 )
          goto LABEL_32;
      }
      if ( (*((_BYTE *)k - 17) & 0x20) == 0 )
        goto LABEL_68;
LABEL_63:
      v44 = sub_B91C10((__int64)(k - 3), 1);
      if ( v44 )
      {
        v45 = sub_B8D060(&v88, v44);
        sub_B99FD0((__int64)(k - 3), 1u, v45);
      }
      v42 = *((_BYTE *)k - 24);
LABEL_66:
      if ( (unsigned __int8)(v42 - 61) <= 1u )
      {
        v89[0] = 0x400000001LL;
        v89[1] = 0x900000007LL;
        v89[2] = 0x110000000BLL;
        v90 = 19;
        sub_B9ADA0((__int64)(k - 3), (unsigned int *)v89, 7);
      }
LABEL_68:
      sub_28FF340((__int64)v89);
      v46 = *((unsigned __int8 *)k - 24);
      if ( (unsigned __int8)(v46 - 34) > 0x33u || (v47 = 0x8000000000041LL, !_bittest64(&v47, (unsigned int)(v46 - 34))) )
      {
LABEL_82:
        v51 = v91;
        while ( v51 )
        {
          v52 = v51;
          sub_28FFC40(*(_QWORD **)(v51 + 24));
          v53 = *(_QWORD *)(v51 + 32);
          v51 = *(_QWORD *)(v51 + 16);
          if ( v53 != v52 + 56 )
            _libc_free(v53);
          j_j___libc_free_0(v52);
        }
        goto LABEL_86;
      }
      if ( v46 == 40 )
      {
        v86 = 32LL * (unsigned int)sub_B491D0((__int64)(k - 3));
      }
      else
      {
        v86 = 0;
        if ( v46 != 85 )
        {
          if ( v46 != 34 )
            BUG();
          v86 = 64;
        }
      }
      if ( *((char *)k - 17) < 0 )
      {
        v55 = sub_BD2BC0((__int64)(k - 3));
        v57 = v55 + v56;
        if ( *((char *)k - 17) >= 0 )
        {
          if ( (unsigned int)(v57 >> 4) )
LABEL_132:
            BUG();
        }
        else if ( (unsigned int)((v57 - sub_BD2BC0((__int64)(k - 3))) >> 4) )
        {
          if ( *((char *)k - 17) >= 0 )
            goto LABEL_132;
          v58 = *(_DWORD *)(sub_BD2BC0((__int64)(k - 3)) + 8);
          if ( *((char *)k - 17) >= 0 )
            BUG();
          v59 = sub_BD2BC0((__int64)(k - 3));
          v61 = 32LL * (unsigned int)(*(_DWORD *)(v59 + v60 - 4) - v58);
          goto LABEL_100;
        }
      }
      v61 = 0;
LABEL_100:
      v62 = (32LL * (*((_DWORD *)k - 5) & 0x7FFFFFF) - 32 - v86 - v61) >> 5;
      if ( (_DWORD)v62 )
      {
        v63 = (unsigned int)(v62 - 1);
        v64 = k + 6;
        v65 = 0;
        v66 = *((_DWORD *)k - 5) & 0x7FFFFFF;
        v79 = v39;
        v67 = k;
        v68 = (__int64)(k - 3);
        v69 = v63;
        v87 = v64;
        while ( 1 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v68 + 32 * (v65 - v66)) + 8LL) + 8LL) == 14 )
          {
            v70 = (__int64 *)sub_BD5C60(v68);
            v67[6] = sub_A7A440(v87, v70, (int)v65 + 1, (__int64)v89);
          }
          if ( v69 == v65 )
            break;
          ++v65;
          v66 = *((_DWORD *)v67 - 5) & 0x7FFFFFF;
        }
        v43 = v68;
        k = v67;
        v39 = v79;
      }
      if ( *(_BYTE *)(*(k - 2) + 8LL) == 14 )
      {
        v50 = (__int64 *)sub_BD5C60(v43);
        k[6] = sub_A7A440(k + 6, v50, 0, (__int64)v89);
      }
      goto LABEL_82;
    }
  }
LABEL_38:
  v93 = (unsigned __int64)v98;
  v94 = 2;
  v96 = 0;
  v97 = 1;
  v99 = 0;
  v100 = v104;
  v101 = 2;
  v102 = 0;
  v103 = 1;
  v95 = 1;
  v98[0] = &unk_4F89C30;
  v92 = 1;
  if ( &unk_4F89C30 != (_UNKNOWN *)&qword_4F82400 && &unk_4F89C30 != &unk_4F6D3F8 )
  {
    v95 = 2;
    v98[1] = &unk_4F6D3F8;
    v92 = 2;
  }
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v98, (__int64)&v92);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v104, (__int64)&v99);
  if ( !v103 )
  {
    _libc_free((unsigned __int64)v100);
    if ( v97 )
      return a1;
    goto LABEL_121;
  }
  if ( !v97 )
LABEL_121:
    _libc_free(v93);
  return a1;
}
