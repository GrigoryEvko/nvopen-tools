// Function: sub_C1B290
// Address: 0xc1b290
//
unsigned __int64 __fastcall sub_C1B290(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // r14
  __int64 *v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // r13
  unsigned __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  char *v16; // r12
  __int64 *v17; // rbx
  __int64 *v18; // rcx
  char *v19; // r13
  char *v20; // r14
  __int64 v21; // r12
  char *v22; // rdx
  __int64 v23; // rax
  char *v24; // r14
  signed __int64 v25; // r9
  char *v26; // rcx
  __int64 v27; // r10
  __int64 i; // rax
  __int64 v29; // rdi
  char *v30; // rdx
  char *v31; // rdi
  char v32; // r8
  char v33; // si
  __int64 v34; // rdx
  char *v35; // rsi
  char *v36; // rdx
  char v37; // r8
  char v38; // r10
  char *v40; // rax
  char *v41; // rdx
  char v42; // si
  char v43; // cl
  unsigned __int64 v44; // r10
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r11
  __int64 v48; // rdi
  unsigned __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rcx
  unsigned __int64 v52; // rsi
  __int64 v53; // rdi
  size_t v54; // rax
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // rax
  _BYTE *v58; // rax
  unsigned __int64 v59; // [rsp+8h] [rbp-188h]
  unsigned __int64 v60; // [rsp+10h] [rbp-180h]
  __int64 v61; // [rsp+20h] [rbp-170h]
  __int64 v62; // [rsp+28h] [rbp-168h]
  __int64 v63; // [rsp+30h] [rbp-160h]
  unsigned __int64 v64; // [rsp+38h] [rbp-158h]
  unsigned __int64 v65; // [rsp+40h] [rbp-150h]
  __int64 v66; // [rsp+48h] [rbp-148h]
  __int64 v67; // [rsp+58h] [rbp-138h]
  __int64 *v68; // [rsp+58h] [rbp-138h]
  char v69; // [rsp+58h] [rbp-138h]
  char v70; // [rsp+58h] [rbp-138h]
  __int64 v71; // [rsp+68h] [rbp-128h]
  __int64 v72; // [rsp+68h] [rbp-128h]
  char *v73; // [rsp+68h] [rbp-128h]
  size_t v74; // [rsp+68h] [rbp-128h]
  _QWORD v75[2]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v76; // [rsp+80h] [rbp-110h] BYREF
  __int64 v77; // [rsp+88h] [rbp-108h]
  __int64 v78; // [rsp+90h] [rbp-100h]
  __int64 v79; // [rsp+98h] [rbp-F8h]
  __int64 v80; // [rsp+A0h] [rbp-F0h]
  __int64 v81; // [rsp+A8h] [rbp-E8h]
  __int64 v82; // [rsp+B0h] [rbp-E0h]
  __int64 v83; // [rsp+B8h] [rbp-D8h]
  char v84[8]; // [rsp+C0h] [rbp-D0h] BYREF
  _BYTE v85[200]; // [rsp+C8h] [rbp-C8h] BYREF

  if ( a1 == a2 )
    return sub_AC25F0(&v76, 0, (__int64)sub_C64CA0);
  v2 = a1;
  v3 = &v76;
  while ( 1 )
  {
    v4 = v2[1];
    v71 = *v2;
    if ( *v2 )
    {
      v67 = v2[1];
      sub_C7D030(v84);
      sub_C7D280(v84, v71, v67);
      sub_C7D290(v84, v75);
      v4 = v75[0];
    }
    ++v3;
    v5 = v4 + 33 * v2[2];
    if ( v3 == (__int64 *)v85 )
      break;
    v2 += 3;
    *(v3 - 1) = v5;
    if ( a2 == v2 )
      return sub_AC25F0(&v76, (char *)v3 - (char *)&v76, (__int64)sub_C64CA0);
  }
  v6 = a2;
  v59 = 64;
  v7 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v8 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v60 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v7
           ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v7
          ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v77 + v8, 27));
  v9 = v80
     - 0x622015F714C7D297LL
     * (((0x9DDFEA08EB382D69LL
        * (v7
         ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (v7
        ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
        ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  v65 = 0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v82 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22)
      + v81
      + v8;
  v10 = v76 - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v11 = 0xB492B66FBE98F273LL
      * __ROL8__(
          v7
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
            ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
            ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
           ^ 0xB492B66FBE98F273LL))),
          31);
  v12 = v79 + v7 + v10;
  v13 = v10 + v78 + v77;
  v63 = v79 + v13;
  v14 = v11 + v9;
  v15 = v14 + v81 + v82;
  v61 = __ROR8__(v60 + v12, 21) + __ROL8__(v13, 20) + v10;
  v66 = v83 + v15;
  v16 = v84;
  v64 = v11;
  v17 = v2;
  v62 = __ROL8__(v15, 20) + __ROR8__(v14 + v65 + v78 + v83, 21) + v14;
  while ( 1 )
  {
    v18 = v6;
    v19 = (char *)&v76;
    v20 = v16;
    while ( 1 )
    {
      v21 = v17[1];
      v72 = *v17;
      if ( *v17 )
      {
        v68 = v18;
        sub_C7D030(v20);
        sub_C7D280(v20, v72, v21);
        sub_C7D290(v20, v75);
        v21 = v75[0];
        v18 = v68;
      }
      v22 = v19 + 8;
      v23 = v21 + 33 * v17[2];
      if ( v19 + 8 == v85 )
      {
        v16 = v20;
        v24 = v19;
        v6 = v18;
        goto LABEL_12;
      }
      v17 += 3;
      *(_QWORD *)v19 = v23;
      if ( v18 == v17 )
        break;
      v19 += 8;
    }
    v16 = v20;
    v6 = v18;
    v24 = v22;
LABEL_12:
    v25 = v24 - (char *)&v76;
    if ( v24 != v16 && v24 != (char *)&v76 )
    {
      if ( v16 - v24 == v25 )
      {
        v40 = (char *)&v76;
        v41 = v24;
        do
        {
          v42 = *v41;
          v43 = *v40++;
          ++v41;
          *(v40 - 1) = v42;
          *(v41 - 1) = v43;
        }
        while ( v40 != v24 );
      }
      else
      {
        v26 = (char *)&v76;
        v27 = v24 - (char *)&v76;
        for ( i = 64; ; i = v29 )
        {
          v29 = i - v27;
          if ( v27 < i - v27 )
            break;
LABEL_23:
          if ( v29 == 1 )
          {
            v69 = v26[i - 1];
            v73 = v26;
            memmove(v26 + 1, v26, i - 1);
            v25 = v24 - (char *)&v76;
            *v73 = v69;
            goto LABEL_36;
          }
          v35 = &v26[i];
          v36 = &v26[i - v29];
          v26 = v36;
          if ( v27 > 0 )
          {
            v26 = &v36[-v27];
            do
            {
              v37 = *(v36 - 1);
              v38 = *(v35 - 1);
              --v36;
              --v35;
              *v36 = v38;
              *v35 = v37;
            }
            while ( v36 != v26 );
          }
          v27 = i % v29;
          if ( !(i % v29) )
            goto LABEL_36;
        }
        while ( v27 != 1 )
        {
          v30 = &v26[v27];
          if ( v29 <= 0 )
          {
            v31 = v26;
          }
          else
          {
            v31 = &v26[v29];
            do
            {
              v32 = *v30;
              v33 = *v26++;
              ++v30;
              *(v26 - 1) = v32;
              *(v30 - 1) = v33;
            }
            while ( v26 != v31 );
          }
          v34 = i % v27;
          if ( !(i % v27) )
            goto LABEL_36;
          i = v27;
          v26 = v31;
          v27 -= v34;
          v29 = i - v27;
          if ( v27 >= i - v27 )
            goto LABEL_23;
        }
        v54 = i - 1;
        if ( v54 )
        {
          v70 = *v26;
          v74 = v54;
          v58 = memmove(v26, v26 + 1, v54);
          v25 = v24 - (char *)&v76;
          v58[v74] = v70;
        }
        else
        {
          *v26 = *v26;
        }
      }
    }
LABEL_36:
    v44 = 0xB492B66FBE98F273LL * __ROL8__(v61 + v65 + v82, 22) + v81 + v63;
    v45 = v76 - 0x4B6D499041670D8DLL * v61;
    v46 = v62 ^ (0xB492B66FBE98F273LL * __ROL8__(v64 + v77 + v63 + v65, 27));
    v65 = v44;
    v47 = 0xB492B66FBE98F273LL * __ROL8__(v66 + v60, 31);
    v64 = v47;
    v48 = v45 + v78 + v77;
    v63 = v79 + v48;
    v61 = __ROR8__(v46 + v45 + v79 + v66, 21) + __ROL8__(v48, 20) + v45;
    v49 = v47 + v62 + v80;
    v50 = __ROR8__(v44 + v49 + v78 + v83, 21);
    v51 = v83 + v49 + v81 + v82;
    v59 += v25;
    v52 = __ROL8__(v49 + v81 + v82, 20) + v49;
    v66 = v51;
    v62 = v50 + v52;
    if ( v6 == v17 )
      break;
    v60 = v46;
  }
  v53 = v50 + v52;
  v55 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v53 ^ v61)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v53 ^ v61)) ^ v53)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v53 ^ v61)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v53 ^ v61)) ^ v53)))
      + v47
      - 0x4B6D499041670D8DLL * (v59 ^ (v59 >> 47));
  v56 = ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v51 ^ v63)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v51 ^ v63)) ^ v51)) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v51 ^ v63)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v51 ^ v63)) ^ v51));
  v57 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v55 ^ (0xB492B66FBE98F273LL * ((v44 >> 47) ^ v44) + v46 - 0x622015F714C7D297LL * v56))) >> 47)
       ^ (0x9DDFEA08EB382D69LL * (v55 ^ (0xB492B66FBE98F273LL * ((v44 >> 47) ^ v44) + v46 - 0x622015F714C7D297LL * v56)))
       ^ v55);
  return 0x9DDFEA08EB382D69LL * ((v57 >> 47) ^ v57);
}
