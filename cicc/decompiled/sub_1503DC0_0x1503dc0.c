// Function: sub_1503DC0
// Address: 0x1503dc0
//
unsigned __int64 *__fastcall sub_1503DC0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v3; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r12
  int v9; // r8d
  unsigned int v10; // edx
  __int64 *v11; // rbx
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r10
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  const char *v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned int v20; // esi
  __int64 v21; // rsi
  unsigned __int64 v22; // rcx
  unsigned __int64 *v23; // r15
  unsigned int v24; // r9d
  __int64 v25; // rsi
  unsigned __int64 v26; // rdi
  __int64 v27; // r8
  char v28; // cl
  unsigned int v29; // r9d
  unsigned int v30; // r9d
  unsigned __int64 v31; // rax
  const char *v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r10
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 j; // r14
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rbx
  __int64 v43; // r13
  __int64 i; // r14
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rdx
  __int64 v49; // rbx
  __int64 *v50; // rcx
  __int64 *v51; // rdi
  __int64 v52; // rax
  __int64 *v53; // r15
  __int64 *v54; // rbx
  __int64 v55; // r12
  __int64 v56; // rdi
  __int64 v57; // rax
  unsigned __int8 v58; // dl
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int64 v61; // rsi
  _QWORD *v62; // rax
  __int64 v63; // rdi
  unsigned __int64 v64; // rsi
  __int64 v65; // rsi
  __int64 *v66; // rdx
  __int64 v67; // r15
  __int64 *v68; // rcx
  __int64 *v69; // rdi
  __int64 v70; // rax
  __int64 *v71; // rbx
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // [rsp+8h] [rbp-2B8h]
  __int64 v75; // [rsp+10h] [rbp-2B0h]
  unsigned __int64 *v76; // [rsp+18h] [rbp-2A8h]
  __int64 v77; // [rsp+20h] [rbp-2A0h]
  __int64 v78; // [rsp+28h] [rbp-298h]
  __int64 v79; // [rsp+30h] [rbp-290h]
  __int64 v80; // [rsp+38h] [rbp-288h]
  __int64 v81; // [rsp+40h] [rbp-280h]
  __int64 v82; // [rsp+48h] [rbp-278h]
  __int64 v83; // [rsp+48h] [rbp-278h]
  __int64 v84; // [rsp+48h] [rbp-278h]
  __int64 v85; // [rsp+58h] [rbp-268h] BYREF
  __int64 v86[2]; // [rsp+60h] [rbp-260h] BYREF
  char v87; // [rsp+70h] [rbp-250h]
  char v88; // [rsp+71h] [rbp-24Fh]
  __int64 v89[2]; // [rsp+80h] [rbp-240h] BYREF
  char v90; // [rsp+90h] [rbp-230h] BYREF
  char v91; // [rsp+91h] [rbp-22Fh]

  v3 = a1;
  if ( *(_BYTE *)(a3 + 16) || (*(_BYTE *)(a3 + 34) & 0x40) == 0 )
  {
    *a1 = 1;
    return v3;
  }
  v6 = *(unsigned int *)(a2 + 1512);
  v7 = *(_QWORD *)(a2 + 1496);
  v8 = a2;
  if ( (_DWORD)v6 )
  {
    v9 = 1;
    v10 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( a3 == *v11 )
      goto LABEL_7;
    while ( v12 != -8 )
    {
      v10 = (v6 - 1) & (v9 + v10);
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( a3 == *v11 )
        goto LABEL_7;
      ++v9;
    }
  }
  v11 = (__int64 *)(v7 + 16 * v6);
LABEL_7:
  while ( !v11[1] )
  {
    v13 = *(_QWORD *)(v8 + 448);
    v14 = *(_QWORD *)(v8 + 40);
    *(_DWORD *)(v8 + 64) = 0;
    v15 = (v13 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 48) = v15;
    v16 = v13 & 0x3F;
    if ( !v16 )
      goto LABEL_9;
    if ( v15 >= v14 )
      goto LABEL_145;
    v22 = v15 + 8;
    v23 = (unsigned __int64 *)(v15 + *(_QWORD *)(v8 + 32));
    if ( v15 + 8 <= v14 )
    {
      v33 = *v23;
      *(_QWORD *)(v8 + 48) = v22;
      *(_QWORD *)(v8 + 56) = v33 >> v16;
      *(_DWORD *)(v8 + 64) = 64 - v16;
      goto LABEL_34;
    }
    *(_QWORD *)(v8 + 56) = 0;
    v24 = v14 - v15;
    if ( (_DWORD)v14 == (_DWORD)v15 )
      goto LABEL_145;
    v25 = 0;
    v26 = 0;
    do
    {
      v27 = *((unsigned __int8 *)v23 + v25);
      v28 = 8 * v25++;
      v26 |= v27 << v28;
      *(_QWORD *)(v8 + 56) = v26;
    }
    while ( v24 != v25 );
    v15 += v24;
    v29 = 8 * v24;
    *(_QWORD *)(v8 + 48) = v15;
    *(_DWORD *)(v8 + 64) = v29;
    if ( v16 > v29 )
LABEL_145:
      sub_16BD130("Unexpected end of file", 1);
    v30 = v29 - v16;
    *(_QWORD *)(v8 + 56) = v26 >> v16;
    *(_DWORD *)(v8 + 64) = v30;
    if ( !v30 )
    {
LABEL_9:
      if ( v15 >= v14 )
      {
        v91 = 1;
        v17 = "Could not find function in stream";
LABEL_11:
        v89[0] = (__int64)v17;
        v90 = 3;
        sub_14EE4B0(&v85, v8 + 8, (__int64)v89);
        v18 = v85 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_12;
      }
    }
LABEL_34:
    if ( !*(_BYTE *)(v8 + 1480) )
    {
      v91 = 1;
      v17 = "Trying to materialize functions before seeing function blocks";
      goto LABEL_11;
    }
    v89[0] = (__int64)&v90;
    v89[1] = 0x4000000000LL;
    v31 = sub_14ECC00(v8 + 32, 0);
    if ( (_DWORD)v31 != 2 )
    {
      v88 = 1;
      v32 = "Expect SubBlock";
      goto LABEL_39;
    }
    if ( HIDWORD(v31) != 12 )
    {
      v88 = 1;
      v32 = "Expect function block";
LABEL_39:
      v86[0] = (__int64)v32;
      v87 = 3;
      sub_14EE4B0(&v85, v8 + 8, (__int64)v86);
      v18 = v85 & 0xFFFFFFFFFFFFFFFELL;
LABEL_12:
      if ( v18 )
        goto LABEL_36;
      continue;
    }
    sub_14F8A10(v86, v8);
    v18 = v86[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v86[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_36;
    *(_QWORD *)(v8 + 448) = 8LL * *(_QWORD *)(v8 + 48) - *(unsigned int *)(v8 + 64);
  }
  sub_14EB830((unsigned __int64 *)v89, v8);
  v18 = v89[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v89[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_36;
  v19 = v11[1];
  *(_DWORD *)(v8 + 64) = 0;
  *(_QWORD *)(v8 + 48) = (v19 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
  v20 = v19 & 0x3F;
  if ( v20 )
    sub_14ECAB0(v8 + 32, v20);
  sub_14FCE40(v89, v8, a3);
  v18 = v89[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v89[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_36:
    *v3 = v18 | 1;
    return v3;
  }
  *(_DWORD *)(a3 + 32) &= ~0x400000u;
  if ( *(_BYTE *)(v8 + 1658) )
    sub_15AC2E0(a3);
  if ( *(_DWORD *)(v8 + 1432) )
  {
    v66 = *(__int64 **)(v8 + 1424);
    v67 = 2LL * *(unsigned int *)(v8 + 1440);
    v68 = &v66[v67];
    v69 = &v66[v67];
    if ( v66 != &v66[v67] )
    {
      while ( 1 )
      {
        v70 = *v66;
        v71 = v66;
        if ( *v66 != -16 && v70 != -8 )
          break;
        v66 += 2;
        if ( v68 == v66 )
          goto LABEL_21;
      }
      if ( v69 != v66 )
      {
        v84 = v8;
        do
        {
          v72 = *(_QWORD *)(v70 + 8);
          while ( v72 )
          {
            while ( 1 )
            {
              v73 = sub_1648700(v72);
              v72 = *(_QWORD *)(v72 + 8);
              if ( *(_BYTE *)(v73 + 16) == 78 )
                break;
              if ( !v72 )
                goto LABEL_137;
            }
            sub_156E800(v73, v71[1]);
          }
LABEL_137:
          v71 += 2;
          if ( v71 == v69 )
            break;
          while ( 1 )
          {
            v70 = *v71;
            if ( *v71 != -8 && v70 != -16 )
              break;
            v71 += 2;
            if ( v69 == v71 )
              goto LABEL_141;
          }
        }
        while ( v69 != v71 );
LABEL_141:
        v8 = v84;
      }
    }
  }
LABEL_21:
  if ( *(_DWORD *)(v8 + 1464) )
  {
    v48 = *(__int64 **)(v8 + 1456);
    v49 = 2LL * *(unsigned int *)(v8 + 1472);
    v50 = &v48[v49];
    v51 = &v48[v49];
    if ( v48 != &v48[v49] )
    {
      while ( 1 )
      {
        v52 = *v48;
        v53 = v48;
        if ( *v48 != -8 && v52 != -16 )
          break;
        v48 += 2;
        if ( v50 == v48 )
          goto LABEL_22;
      }
      if ( v48 != v51 )
      {
        v83 = v8;
        v54 = v51;
        do
        {
          v55 = *(_QWORD *)(v52 + 8);
LABEL_102:
          if ( v55 )
          {
            while ( 1 )
            {
              v56 = v55;
              v55 = *(_QWORD *)(v55 + 8);
              v57 = sub_1648700(v56);
              v58 = *(_BYTE *)(v57 + 16);
              if ( v58 <= 0x17u )
                break;
              if ( v58 == 78 )
              {
                v59 = v57 | 4;
              }
              else
              {
                v59 = v57 & 0xFFFFFFFFFFFFFFFBLL;
                if ( v58 != 29 )
                  break;
              }
              v60 = v53[1];
              v61 = v59 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v59 & 4) != 0 )
              {
                v62 = (_QWORD *)(v61 - 24);
                goto LABEL_108;
              }
LABEL_107:
              v62 = (_QWORD *)(v61 - 72);
LABEL_108:
              if ( *v62 )
              {
                v63 = v62[1];
                v64 = v62[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v64 = v63;
                if ( v63 )
                  *(_QWORD *)(v63 + 16) = *(_QWORD *)(v63 + 16) & 3LL | v64;
              }
              *v62 = v60;
              if ( !v60 )
                goto LABEL_102;
              v65 = *(_QWORD *)(v60 + 8);
              v62[1] = v65;
              if ( v65 )
                *(_QWORD *)(v65 + 16) = (unsigned __int64)(v62 + 1) | *(_QWORD *)(v65 + 16) & 3LL;
              v62[2] = (v60 + 8) | v62[2] & 3LL;
              *(_QWORD *)(v60 + 8) = v62;
              if ( !v55 )
                goto LABEL_115;
            }
            v60 = v53[1];
            v61 = 0;
            goto LABEL_107;
          }
LABEL_115:
          v53 += 2;
          if ( v53 == v54 )
            break;
          while ( 1 )
          {
            v52 = *v53;
            if ( *v53 != -16 && v52 != -8 )
              break;
            v53 += 2;
            if ( v54 == v53 )
              goto LABEL_119;
          }
        }
        while ( v54 != v53 );
LABEL_119:
        v8 = v83;
      }
    }
  }
LABEL_22:
  v80 = v8 + 608;
  v21 = sub_15160C0(v8 + 608, a3);
  if ( v21 )
    sub_1627150(a3, v21);
  if ( !(unsigned __int8)sub_1516160(v80, v21) )
  {
    v34 = *(_QWORD *)(a3 + 80);
    v81 = a3 + 72;
    if ( a3 + 72 == v34 )
    {
      v35 = 0;
    }
    else
    {
      do
      {
        if ( !v34 )
          goto LABEL_146;
        v35 = *(_QWORD *)(v34 + 24);
        if ( v35 != v34 + 16 )
          break;
        v34 = *(_QWORD *)(v34 + 8);
      }
      while ( a3 + 72 != v34 );
    }
    v76 = v3;
    v79 = v8 + 1664;
    v75 = v8;
    v36 = v34;
    v74 = a3;
    v37 = v35;
LABEL_53:
    if ( v36 != v81 )
    {
      j = v37;
      do
      {
        if ( !j )
          goto LABEL_146;
        if ( *(_QWORD *)(j + 24) || *(__int16 *)(j - 6) < 0 )
        {
          v39 = sub_1625790(j - 24, 1);
          if ( v39 )
          {
            if ( !(unsigned __int8)sub_16635B0(v79, j - 24, v39) )
            {
              sub_1516150(v80, 1);
              v40 = *(_QWORD *)(v74 + 40);
              v82 = v40 + 24;
              if ( *(_QWORD *)(v40 + 32) != v40 + 24 )
              {
                v78 = v36;
                v41 = *(_QWORD *)(v40 + 32);
                v77 = j;
                do
                {
                  if ( !v41 )
                    BUG();
                  if ( (*(_BYTE *)(v41 - 22) & 0x40) == 0 )
                  {
                    v42 = *(_QWORD *)(v41 + 24);
                    v43 = v41 + 16;
                    if ( v41 + 16 != v42 )
                    {
                      while ( v42 )
                      {
                        i = *(_QWORD *)(v42 + 24);
                        if ( i == v42 + 16 )
                        {
                          v42 = *(_QWORD *)(v42 + 8);
                          if ( v43 != v42 )
                            continue;
                        }
                        goto LABEL_68;
                      }
LABEL_146:
                      BUG();
                    }
                    i = 0;
LABEL_68:
                    while ( v43 != v42 )
                    {
                      v45 = i - 24;
                      if ( !i )
                        v45 = 0;
                      sub_1625C10(v45, 1, 0);
                      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v42 + 24) )
                      {
                        v46 = v42 - 24;
                        if ( !v42 )
                          v46 = 0;
                        if ( i != v46 + 40 )
                          break;
                        v42 = *(_QWORD *)(v42 + 8);
                        if ( v43 == v42 )
                          goto LABEL_68;
                        if ( !v42 )
                          goto LABEL_146;
                      }
                    }
                  }
                  v41 = *(_QWORD *)(v41 + 8);
                }
                while ( v82 != v41 );
                v36 = v78;
                j = v77;
              }
            }
          }
        }
        for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v36 + 24) )
        {
          v47 = v36 - 24;
          if ( !v36 )
            v47 = 0;
          if ( j != v47 + 40 )
            break;
          v36 = *(_QWORD *)(v36 + 8);
          if ( v81 == v36 )
          {
            v37 = j;
            goto LABEL_53;
          }
          if ( !v36 )
            goto LABEL_146;
        }
      }
      while ( v81 != v36 );
    }
    v3 = v76;
    v8 = v75;
  }
  sub_15046F0(v3, v8);
  return v3;
}
