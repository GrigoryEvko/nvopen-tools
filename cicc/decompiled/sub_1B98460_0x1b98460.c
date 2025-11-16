// Function: sub_1B98460
// Address: 0x1b98460
//
unsigned __int64 __fastcall sub_1B98460(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v3; // r9
  __int64 *v4; // rbx
  __int64 *v5; // rsi
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  unsigned __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r10
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // r11
  unsigned __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rdx
  char *i; // r12
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v28; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v29; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v30; // [rsp+30h] [rbp-A0h]
  __int64 v31; // [rsp+38h] [rbp-98h]
  unsigned __int64 v32; // [rsp+40h] [rbp-90h]
  unsigned __int64 v33; // [rsp+48h] [rbp-88h]
  unsigned __int64 v34; // [rsp+50h] [rbp-80h]
  __int64 v35; // [rsp+60h] [rbp-70h] BYREF
  __int64 v36; // [rsp+68h] [rbp-68h] BYREF
  __int64 v37; // [rsp+70h] [rbp-60h]
  __int64 v38; // [rsp+78h] [rbp-58h]
  __int64 v39; // [rsp+80h] [rbp-50h]
  __int64 v40; // [rsp+88h] [rbp-48h]
  __int64 v41; // [rsp+90h] [rbp-40h]
  __int64 v42; // [rsp+98h] [rbp-38h]
  char v43[8]; // [rsp+A0h] [rbp-30h] BYREF
  _BYTE v44[40]; // [rsp+A8h] [rbp-28h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v26 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v26 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v26;
    sub_2207640(byte_4F99930);
  }
  v3 = qword_4F99938;
  if ( a2 == a1 )
  {
    v7 = 0;
    return sub_1593600(&v35, v7, v3);
  }
  v4 = a1 + 3;
  v5 = &v36;
  v35 = *a1;
  if ( a1 + 3 == a2 )
  {
LABEL_6:
    v7 = (char *)v5 - (char *)&v35;
    return sub_1593600(&v35, v7, v3);
  }
  while ( v44 != (_BYTE *)++v5 )
  {
    v6 = *v4;
    v4 += 3;
    *(v5 - 1) = v6;
    if ( v4 == a2 )
      goto LABEL_6;
  }
  v28 = 0;
  v29 = v3;
  v31 = __ROL8__(v3 ^ 0xB492B66FBE98F273LL, 15);
  v33 = v3 ^ (v3 >> 47);
  v30 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v3 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v3 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v3 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v3 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v32 = 0xB492B66FBE98F273LL * v3;
  v34 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v33 ^ (0xB492B66FBE98F273LL * v3)))
          ^ v33
          ^ ((0x9DDFEA08EB382D69LL * (v33 ^ (0xB492B66FBE98F273LL * v3))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v33 ^ (0xB492B66FBE98F273LL * v3)))
         ^ v33
         ^ ((0x9DDFEA08EB382D69LL * (v33 ^ (0xB492B66FBE98F273LL * v3))) >> 47))));
  sub_1593A20(&v28, &v35);
  v27 = 64;
  do
  {
    v21 = *v4;
    for ( i = (char *)&v36; ; i += 8 )
    {
      v4 += 3;
      *((_QWORD *)i - 1) = v21;
      if ( a2 == v4 || i + 8 == v44 )
        break;
      v21 = *v4;
    }
    sub_1B8E7D0((char *)&v35, i, v43);
    v9 = v30;
    v10 = v35 - 0x4B6D499041670D8DLL * v32;
    v11 = v34 ^ (0xB492B66FBE98F273LL * __ROL8__(v36 + v31 + v29 + v28, 27));
    v12 = v40 + v31 - 0x4B6D499041670D8DLL * __ROL8__(v41 + v32 + v29, 22);
    v30 = v11;
    v29 = v12;
    v13 = v10 + v37 + v36;
    v14 = v38 + v13;
    v15 = 0xB492B66FBE98F273LL * __ROL8__(v33 + v9, 31);
    v31 = v38 + v13;
    v16 = v15 + v39 + v34;
    v17 = __ROL8__(v13, 20) + v10 + __ROR8__(v11 + v10 + v38 + v33, 21);
    v28 = v15;
    v32 = v17;
    v18 = v16 + v40 + v41;
    v19 = v42 + v18;
    v27 += i - (char *)&v35;
    v33 = v42 + v18;
    v20 = __ROR8__(v12 + v16 + v37 + v42, 21) + __ROL8__(v18, 20) + v16;
    v34 = v20;
  }
  while ( a2 != v4 );
  v23 = 0xB492B66FBE98F273LL * (v27 ^ (v27 >> 47))
      + v15
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v20 ^ v17)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v17)) ^ v20)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v20 ^ v17)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v20 ^ v17)) ^ v20)));
  v24 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v19 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v19 ^ v14)) ^ v19)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v19 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v19 ^ v14)) ^ v19)));
  v25 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v23 ^ (0xB492B66FBE98F273LL * ((v12 >> 47) ^ v12) + v11 + v24))) >> 47)
       ^ v23
       ^ (0x9DDFEA08EB382D69LL * (v23 ^ (0xB492B66FBE98F273LL * ((v12 >> 47) ^ v12) + v11 + v24))));
  return 0x9DDFEA08EB382D69LL * ((v25 >> 47) ^ v25);
}
