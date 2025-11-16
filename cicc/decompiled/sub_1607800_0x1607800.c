// Function: sub_1607800
// Address: 0x1607800
//
unsigned __int64 __fastcall sub_1607800(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v4; // r11
  unsigned __int64 v5; // r8
  unsigned __int64 v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // r13
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r14
  __int64 v15; // r8
  unsigned __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // r9
  unsigned __int64 v20; // r14
  __int64 v21; // rdi
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rcx
  _QWORD *i; // r8
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int64 v35; // r12
  __int64 v36; // rcx
  __int64 v37; // r15
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // r14
  unsigned __int64 v40; // r9
  unsigned __int64 v41; // rax
  __int64 v42; // r10
  unsigned __int64 v43; // r8
  __int64 v44; // r13
  unsigned __int64 v45; // r14
  __int64 v46; // rdx
  __int64 v47; // r12
  unsigned __int64 v48; // r8
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdi
  unsigned __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // r10
  __int64 v55; // rcx

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v7 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v7 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v7;
    sub_2207640(byte_4F99930);
    v5 = qword_4F99938;
    v4 = a2 - (_QWORD)a1;
    if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
      goto LABEL_8;
    return sub_1593600(a1, v4, v5);
  }
  v4 = a2 - (_QWORD)a1;
  v5 = qword_4F99938;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_1593600(a1, v4, v5);
LABEL_8:
  v8 = __ROL8__(v5 ^ 0xB492B66FBE98F273LL, 15);
  v9 = (_QWORD *)((char *)a1 + (v4 & 0xFFFFFFFFFFFFFFC0LL));
  v10 = v5 ^ (v5 >> 47);
  v11 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (v10
          ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * v5)))
          ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * v5))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (v10
         ^ (0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * v5)))
         ^ ((0x9DDFEA08EB382D69LL * (v10 ^ (0xB492B66FBE98F273LL * v5))) >> 47))));
  v12 = a1[5] + v8 - 0x4B6D499041670D8DLL * __ROL8__(a1[6] - 0x4B6D499041670D8CLL * v5, 22);
  v13 = v11 ^ (0xB492B66FBE98F273LL * __ROL8__(a1[1] + v5 + v8, 27));
  v14 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v15 = *a1 - 0x6D8ED9027DD26057LL * v5;
  v16 = a1[3] + v10;
  v17 = __ROL8__(v10 + v14, 31);
  v18 = v15 + a1[2] + a1[1];
  v19 = a1[3] + v18;
  v20 = 0xB492B66FBE98F273LL * v17;
  v21 = __ROL8__(v18, 20) + v15 + __ROR8__(v13 + v15 + v16, 21);
  v22 = a1[4] + v11 + v20;
  v23 = v22 + a1[5] + a1[6];
  v24 = a1[7] + v23;
  v25 = v22 + __ROL8__(v23, 20) + __ROR8__(v22 + v12 + a1[2] + a1[7], 21);
  for ( i = a1 + 8; v9 != i; v25 = v37 + v35 + v36 )
  {
    v27 = v13;
    v28 = v12 + v19;
    v29 = __ROL8__(i[6] + v21 + v12, 22);
    v30 = *i - 0x4B6D499041670D8DLL * v21;
    v31 = __ROL8__(v20 + i[1] + v28, 27);
    v12 = i[5] + v19 - 0x4B6D499041670D8DLL * v29;
    v20 = 0xB492B66FBE98F273LL * __ROL8__(v27 + v24, 31);
    v13 = v25 ^ (0xB492B66FBE98F273LL * v31);
    v32 = v30 + i[2] + i[1];
    v19 = i[3] + v32;
    v33 = __ROR8__(v13 + v30 + i[3] + v24, 21);
    v34 = __ROL8__(v32, 20) + v30;
    v35 = i[4] + v25 + v20;
    v21 = v33 + v34;
    v36 = __ROR8__(v12 + v35 + i[2] + i[7], 21);
    v37 = __ROL8__(v35 + i[5] + i[6], 20);
    v24 = i[7] + v35 + i[5] + i[6];
    i += 8;
  }
  if ( (v4 & 0x3F) != 0 )
  {
    v42 = *(_QWORD *)(a2 - 16);
    v43 = v20 + v12;
    v44 = *(_QWORD *)(a2 - 24);
    v45 = v13 + v24;
    v46 = *(_QWORD *)(a2 - 48);
    v20 = 0xB492B66FBE98F273LL * __ROL8__(v45, 31);
    v12 = v44 + v19 - 0x4B6D499041670D8DLL * __ROL8__(v42 + v21 + v12, 22);
    v47 = *(_QWORD *)(a2 - 64) - 0x4B6D499041670D8DLL * v21;
    v48 = v25 ^ (0xB492B66FBE98F273LL * __ROL8__(v19 + *(_QWORD *)(a2 - 56) + v43, 27));
    v49 = v47 + v46 + *(_QWORD *)(a2 - 56);
    v50 = __ROR8__(v47 + *(_QWORD *)(a2 - 40) + v24 + v48, 21);
    v19 = *(_QWORD *)(a2 - 40) + v49;
    v51 = v47 + __ROL8__(v49, 20);
    v52 = *(_QWORD *)(a2 - 32) + v25 + v20;
    v21 = v50 + v51;
    v53 = *(_QWORD *)(a2 - 8);
    v54 = v52 + v44 + v42;
    v55 = v53 + v46;
    v13 = v48;
    v25 = __ROL8__(v54, 20) + v52 + __ROR8__(v12 + v52 + v55, 21);
    v24 = v54 + v53;
  }
  v38 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v25 ^ v21)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v25 ^ v21)) ^ v25);
  v39 = 0x9DDFEA08EB382D69LL * ((v38 >> 47) ^ v38) + 0xB492B66FBE98F273LL * (v4 ^ (v4 >> 47)) + v20;
  v40 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v24 ^ v19)) >> 47) ^ v24 ^ (0x9DDFEA08EB382D69LL * (v24 ^ v19)));
  v41 = 0x9DDFEA08EB382D69LL
      * (v39 ^ (v13 - 0x4B6D499041670D8DLL * ((v12 >> 47) ^ v12) - 0x622015F714C7D297LL * ((v40 >> 47) ^ v40)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v41 ^ v39 ^ (v41 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v41 ^ v39 ^ (v41 >> 47))));
}
