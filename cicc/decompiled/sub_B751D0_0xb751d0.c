// Function: sub_B751D0
// Address: 0xb751d0
//
unsigned __int64 __fastcall sub_B751D0(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 *v3; // rcx
  __int64 v4; // rsi
  __int64 *v5; // r15
  char *v6; // r9
  unsigned __int64 v7; // rdi
  __int64 v8; // r11
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rdx
  __int64 v18; // r10
  char *i; // rax
  __int64 *v20; // r13
  __int64 v21; // rdi
  char *v22; // rbx
  __int64 j; // rax
  __int64 v24; // rsi
  char *v25; // rdx
  char *v26; // rsi
  char v27; // r8
  char v28; // r11
  __int64 v29; // rdx
  char *v30; // r11
  char *v31; // rdx
  char v32; // di
  char v33; // r8
  char *v34; // rdx
  char *v35; // rsi
  char v36; // r11
  char v37; // di
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // r10
  unsigned __int64 v43; // rsi
  size_t v44; // rax
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdx
  char *v48; // [rsp+8h] [rbp-D8h]
  __int64 v49; // [rsp+10h] [rbp-D0h]
  __int64 *v50; // [rsp+10h] [rbp-D0h]
  __int64 *v51; // [rsp+18h] [rbp-C8h]
  __int64 v52; // [rsp+18h] [rbp-C8h]
  char *v53; // [rsp+20h] [rbp-C0h]
  char v54; // [rsp+20h] [rbp-C0h]
  char v55; // [rsp+28h] [rbp-B8h]
  size_t v56; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v57; // [rsp+38h] [rbp-A8h]
  __int64 v58; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v59; // [rsp+50h] [rbp-90h]
  __int64 v60; // [rsp+58h] [rbp-88h]
  __int64 v61; // [rsp+60h] [rbp-80h]
  unsigned __int64 v62; // [rsp+68h] [rbp-78h]
  __int64 v63; // [rsp+70h] [rbp-70h] BYREF
  __int64 v64; // [rsp+78h] [rbp-68h] BYREF
  __int64 v65; // [rsp+80h] [rbp-60h]
  __int64 v66; // [rsp+88h] [rbp-58h]
  __int64 v67; // [rsp+90h] [rbp-50h]
  __int64 v68; // [rsp+98h] [rbp-48h]
  __int64 v69; // [rsp+A0h] [rbp-40h]
  __int64 v70; // [rsp+A8h] [rbp-38h]
  char v71; // [rsp+B0h] [rbp-30h] BYREF
  _BYTE v72[40]; // [rsp+B8h] [rbp-28h] BYREF

  if ( a1 == a2 )
    return sub_AC25F0(&v63, 0, (__int64)sub_C64CA0);
  v2 = &v64;
  v3 = a2;
  v4 = *a1;
  v5 = a1;
  v6 = v72;
  do
  {
    ++v5;
    *(v2 - 1) = v4;
    if ( v3 == v5 )
      return sub_AC25F0(&v63, (char *)v2 - (char *)&v63, (__int64)sub_C64CA0);
    ++v2;
    v4 = *v5;
  }
  while ( v2 != (__int64 *)v72 );
  v7 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v8 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v57 = 64;
  v59 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v7
           ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
           ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v7
          ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__((char *)sub_C64CA0 + v64 + v8, 27));
  v9 = v68
     + v8
     - 0x4B6D499041670D8DLL * __ROL8__((char *)sub_C64CA0 + v69 + 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0, 22);
  v10 = v67
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v7
          ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v7
         ^ (0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ ((0x9DDFEA08EB382D69LL * (v7 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  v11 = v63 - 0x6D8ED9027DD26057LL * (_QWORD)sub_C64CA0;
  v12 = 0xB492B66FBE98F273LL
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
  v13 = v66 + v7 + v11;
  v14 = v11 + v65 + v64;
  v60 = v66 + v14;
  v15 = v12 + v10;
  v61 = __ROR8__(v59 + v13, 21) + __ROL8__(v14, 20) + v11;
  v16 = v15 + v68 + v69;
  v17 = __ROL8__(v16, 20) + __ROR8__(v15 + v9 + v65 + v70, 21);
  v18 = v70 + v16;
  v62 = v17 + v15;
  while ( 1 )
  {
    for ( i = (char *)&v64; ; i += 8 )
    {
      v20 = v5 + 1;
      *((_QWORD *)i - 1) = v4;
      if ( v3 == v5 + 1 || i + 8 == v6 )
        break;
      v4 = *v20;
      ++v5;
    }
    v58 = i - (char *)&v63;
    if ( i != &v71 )
    {
      if ( &v71 - i == i - (char *)&v63 )
      {
        v34 = (char *)&v63;
        v35 = i;
        do
        {
          v36 = *v35;
          v37 = *v34++;
          ++v35;
          *(v34 - 1) = v36;
          *(v35 - 1) = v37;
        }
        while ( v34 != i );
      }
      else
      {
        v21 = i - (char *)&v63;
        v22 = (char *)&v63;
        for ( j = 64; ; j = v24 )
        {
          v24 = j - v21;
          if ( v21 < j - v21 )
            break;
LABEL_19:
          if ( v24 == 1 )
          {
            v49 = v18;
            v51 = v3;
            v53 = v6;
            v55 = v22[j - 1];
            memmove(v22 + 1, v22, j - 1);
            v6 = v53;
            v3 = v51;
            v18 = v49;
            *v22 = v55;
            goto LABEL_28;
          }
          v30 = &v22[j];
          v31 = &v22[j - v24];
          v22 = v31;
          if ( v21 > 0 )
          {
            v22 = &v31[-v21];
            do
            {
              v32 = *(v31 - 1);
              v33 = *(v30 - 1);
              --v31;
              --v30;
              *v31 = v33;
              *v30 = v32;
            }
            while ( v31 != v22 );
          }
          v21 = j % v24;
          if ( !(j % v24) )
            goto LABEL_28;
        }
        while ( v21 != 1 )
        {
          v25 = &v22[v21];
          if ( v24 <= 0 )
          {
            v26 = v22;
          }
          else
          {
            v26 = &v22[v24];
            do
            {
              v27 = *v25;
              v28 = *v22++;
              ++v25;
              *(v22 - 1) = v27;
              *(v25 - 1) = v28;
            }
            while ( v22 != v26 );
          }
          v29 = j % v21;
          if ( !(j % v21) )
            goto LABEL_28;
          j = v21;
          v22 = v26;
          v21 -= v29;
          v24 = j - v21;
          if ( v21 >= j - v21 )
            goto LABEL_19;
        }
        v44 = j - 1;
        if ( v44 )
        {
          v48 = v6;
          v50 = v3;
          v52 = v18;
          v54 = *v22;
          v56 = v44;
          memmove(v22, v22 + 1, v44);
          v6 = v48;
          v3 = v50;
          v22[v56] = v54;
          v18 = v52;
        }
        else
        {
          *v22 = *v22;
        }
      }
    }
LABEL_28:
    v38 = __ROL8__(v12 + v64 + v9 + v60, 27);
    v9 = v68 + v60 - 0x4B6D499041670D8DLL * __ROL8__(v69 + v61 + v9, 22);
    v39 = v63 - 0x4B6D499041670D8DLL * v61;
    v40 = v62 ^ (0xB492B66FBE98F273LL * v38);
    v12 = 0xB492B66FBE98F273LL * __ROL8__(v18 + v59, 31);
    v41 = v39 + v18 + v66;
    v42 = v39 + v65 + v64;
    v60 = v66 + v42;
    v61 = __ROL8__(v42, 20) + v39 + __ROR8__(v40 + v41, 21);
    v43 = v67 + v62 + v12;
    v18 = v70 + v43 + v68 + v69;
    v57 += v58;
    v62 = __ROR8__(v9 + v43 + v65 + v70, 21) + __ROL8__(v43 + v68 + v69, 20) + v43;
    if ( v3 == v20 )
      break;
    v4 = v5[1];
    v59 = v40;
    ++v5;
  }
  v45 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v62 ^ v61)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v62 ^ v61)) ^ v62)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v62 ^ v61)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v62 ^ v61)) ^ v62)))
      + v12
      - 0x4B6D499041670D8DLL * (v57 ^ (v57 >> 47));
  v46 = 0xB492B66FBE98F273LL * ((v9 >> 47) ^ v9)
      + v40
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v18 ^ v60)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v60)) ^ v18)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v18 ^ v60)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v18 ^ v60)) ^ v18)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v45 ^ v46)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v46)) ^ v45)) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v45 ^ v46)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v45 ^ v46)) ^ v45)));
}
