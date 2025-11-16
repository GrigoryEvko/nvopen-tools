// Function: sub_1607160
// Address: 0x1607160
//
unsigned __int64 __fastcall sub_1607160(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // rcx
  __int64 *v3; // r15
  unsigned __int64 v4; // r13
  __int64 *v5; // rax
  __int64 v6; // rsi
  char *v7; // r9
  unsigned __int64 v8; // rdi
  __int64 v9; // r11
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // rdx
  __int64 v20; // r10
  char *i; // rax
  __int64 *v22; // r13
  __int64 v23; // rdi
  char *v24; // rbx
  __int64 j; // rax
  __int64 v26; // rsi
  char *v27; // rdx
  char *v28; // rsi
  char v29; // r8
  char v30; // r11
  __int64 v31; // rdx
  char *v32; // r11
  char *v33; // rdx
  char v34; // di
  char v35; // r8
  char *v36; // rdx
  char *v37; // rsi
  char v38; // r11
  char v39; // di
  __int64 v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // r10
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rdi
  size_t v47; // rax
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdx
  int v51; // eax
  unsigned __int64 v52; // rax
  char *v53; // [rsp+8h] [rbp-D8h]
  __int64 v54; // [rsp+10h] [rbp-D0h]
  __int64 *v55; // [rsp+10h] [rbp-D0h]
  __int64 *v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  char *v58; // [rsp+20h] [rbp-C0h]
  char v59; // [rsp+20h] [rbp-C0h]
  char v60; // [rsp+28h] [rbp-B8h]
  size_t v61; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v62; // [rsp+38h] [rbp-A8h]
  __int64 v63; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v64; // [rsp+50h] [rbp-90h]
  __int64 v65; // [rsp+58h] [rbp-88h]
  unsigned __int64 v66; // [rsp+60h] [rbp-80h]
  __int64 v67; // [rsp+68h] [rbp-78h]
  __int64 v68; // [rsp+70h] [rbp-70h] BYREF
  __int64 v69; // [rsp+78h] [rbp-68h] BYREF
  __int64 v70; // [rsp+80h] [rbp-60h]
  __int64 v71; // [rsp+88h] [rbp-58h]
  __int64 v72; // [rsp+90h] [rbp-50h]
  __int64 v73; // [rsp+98h] [rbp-48h]
  __int64 v74; // [rsp+A0h] [rbp-40h]
  __int64 v75; // [rsp+A8h] [rbp-38h]
  char v76; // [rsp+B0h] [rbp-30h] BYREF
  _BYTE v77[40]; // [rsp+B8h] [rbp-28h] BYREF

  v2 = a2;
  v3 = a1;
  if ( !byte_4F99930[0] )
  {
    v51 = sub_2207590(byte_4F99930);
    v2 = a2;
    if ( v51 )
    {
      v52 = unk_4FA04C8;
      if ( !unk_4FA04C8 )
        v52 = 0xFF51AFD7ED558CCDLL;
      qword_4F99938 = v52;
      sub_2207640(byte_4F99930);
      v2 = a2;
    }
  }
  v4 = qword_4F99938;
  if ( a1 == v2 )
    return sub_1593600(&v68, 0, qword_4F99938);
  v5 = &v69;
  v6 = *a1;
  v7 = v77;
  do
  {
    ++v3;
    *(v5 - 1) = v6;
    if ( v2 == v3 )
      return sub_1593600(&v68, (char *)v5 - (char *)&v68, v4);
    ++v5;
    v6 = *v3;
  }
  while ( v5 != (__int64 *)v77 );
  v8 = v4 ^ (v4 >> 47);
  v9 = __ROL8__(v4 ^ 0xB492B66FBE98F273LL, 15);
  v62 = 64;
  v64 = (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v8
           ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
           ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47)))))
      ^ (0xB492B66FBE98F273LL * __ROL8__(v9 + v4 + v69, 27));
  v10 = v73 + v9 - 0x4B6D499041670D8DLL * __ROL8__(v4 + v74 - 0x4B6D499041670D8DLL * v4, 22);
  v11 = v8
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v12 = v72
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (v8
          ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
          ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v8
         ^ (0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4)))
         ^ ((0x9DDFEA08EB382D69LL * (v8 ^ (0xB492B66FBE98F273LL * v4))) >> 47))));
  v13 = v68 - 0x6D8ED9027DD26057LL * v4;
  v14 = 0xB492B66FBE98F273LL * __ROL8__(v11, 31);
  v15 = v71 + v8 + v13;
  v16 = v13 + v70 + v69;
  v65 = v71 + v16;
  v17 = v14 + v12;
  v67 = __ROR8__(v64 + v15, 21) + __ROL8__(v16, 20) + v13;
  v18 = v17 + v73 + v74;
  v19 = __ROL8__(v18, 20) + __ROR8__(v17 + v10 + v70 + v75, 21);
  v20 = v75 + v18;
  v66 = v19 + v17;
  while ( 1 )
  {
    for ( i = (char *)&v69; ; i += 8 )
    {
      v22 = v3 + 1;
      *((_QWORD *)i - 1) = v6;
      if ( v2 == v3 + 1 || i + 8 == v7 )
        break;
      v6 = *v22;
      ++v3;
    }
    v63 = i - (char *)&v68;
    if ( i != &v76 )
    {
      if ( &v76 - i == i - (char *)&v68 )
      {
        v36 = (char *)&v68;
        v37 = i;
        do
        {
          v38 = *v37;
          v39 = *v36++;
          ++v37;
          *(v36 - 1) = v38;
          *(v37 - 1) = v39;
        }
        while ( v36 != i );
      }
      else
      {
        v23 = i - (char *)&v68;
        v24 = (char *)&v68;
        for ( j = 64; ; j = v26 )
        {
          v26 = j - v23;
          if ( v23 < j - v23 )
            break;
LABEL_20:
          if ( v26 == 1 )
          {
            v54 = v20;
            v56 = v2;
            v58 = v7;
            v60 = v24[j - 1];
            memmove(v24 + 1, v24, j - 1);
            v7 = v58;
            v2 = v56;
            v20 = v54;
            *v24 = v60;
            goto LABEL_29;
          }
          v32 = &v24[j];
          v33 = &v24[j - v26];
          v24 = v33;
          if ( v23 > 0 )
          {
            v24 = &v33[-v23];
            do
            {
              v34 = *(v33 - 1);
              v35 = *(v32 - 1);
              --v33;
              --v32;
              *v33 = v35;
              *v32 = v34;
            }
            while ( v33 != v24 );
          }
          v23 = j % v26;
          if ( !(j % v26) )
            goto LABEL_29;
        }
        while ( v23 != 1 )
        {
          v27 = &v24[v23];
          if ( v26 <= 0 )
          {
            v28 = v24;
          }
          else
          {
            v28 = &v24[v26];
            do
            {
              v29 = *v27;
              v30 = *v24++;
              ++v27;
              *(v24 - 1) = v29;
              *(v27 - 1) = v30;
            }
            while ( v24 != v28 );
          }
          v31 = j % v23;
          if ( !(j % v23) )
            goto LABEL_29;
          j = v23;
          v24 = v28;
          v23 -= v31;
          v26 = j - v23;
          if ( v23 >= j - v23 )
            goto LABEL_20;
        }
        v47 = j - 1;
        if ( v47 )
        {
          v53 = v7;
          v55 = v2;
          v57 = v20;
          v59 = *v24;
          v61 = v47;
          memmove(v24, v24 + 1, v47);
          v7 = v53;
          v2 = v55;
          v24[v61] = v59;
          v20 = v57;
        }
        else
        {
          *v24 = *v24;
        }
      }
    }
LABEL_29:
    v40 = __ROL8__(v14 + v69 + v10 + v65, 27);
    v10 = v73 + v65 - 0x4B6D499041670D8DLL * __ROL8__(v74 + v67 + v10, 22);
    v41 = v68 - 0x4B6D499041670D8DLL * v67;
    v42 = v66 ^ (0xB492B66FBE98F273LL * v40);
    v14 = 0xB492B66FBE98F273LL * __ROL8__(v20 + v64, 31);
    v43 = v41 + v20 + v71;
    v44 = v41 + v70 + v69;
    v65 = v71 + v44;
    v67 = __ROL8__(v44, 20) + v41 + __ROR8__(v42 + v43, 21);
    v45 = v14 + v66 + v72;
    v46 = v45 + v73 + v74;
    v20 = v75 + v46;
    v62 += v63;
    v66 = __ROR8__(v10 + v45 + v70 + v75, 21) + __ROL8__(v46, 20) + v45;
    if ( v2 == v22 )
      break;
    v6 = v3[1];
    v64 = v42;
    ++v3;
  }
  v48 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v66 ^ v67)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v66 ^ v67)) ^ v66)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v66 ^ v67)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v66 ^ v67)) ^ v66)))
      + v14
      - 0x4B6D499041670D8DLL * (v62 ^ (v62 >> 47));
  v49 = 0xB492B66FBE98F273LL * ((v10 >> 47) ^ v10)
      + v42
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v20 ^ v65)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v65)) ^ v20)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v20 ^ v65)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v65)) ^ v20)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v48 ^ v49)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v48 ^ v49)) ^ v48)) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v48 ^ v49)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v48 ^ v49)) ^ v48)));
}
