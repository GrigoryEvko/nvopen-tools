// Function: sub_1930F10
// Address: 0x1930f10
//
unsigned __int64 __fastcall sub_1930F10(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // r8
  unsigned __int64 v7; // rax
  _QWORD *v8; // r14
  unsigned __int64 v9; // rax
  _QWORD *v10; // rdi
  unsigned __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r10
  __int64 v14; // r8
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r11
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // r15
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // r13
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v30; // [rsp+18h] [rbp-68h]
  unsigned __int64 v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  unsigned __int64 v33; // [rsp+30h] [rbp-50h]
  unsigned __int64 v34; // [rsp+38h] [rbp-48h]
  unsigned __int64 v35; // [rsp+40h] [rbp-40h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v7 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v7 = 0xFF51AFD7ED558CCDLL;
    v4 = a2 - (_QWORD)a1;
    qword_4F99938 = v7;
    sub_2207640(byte_4F99930);
    v5 = qword_4F99938;
    if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
      goto LABEL_8;
    return sub_1593600(a1, v4, v5);
  }
  v4 = a2 - (_QWORD)a1;
  v5 = qword_4F99938;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_1593600(a1, v4, v5);
LABEL_8:
  v30 = v5;
  v29 = 0;
  v8 = (_QWORD *)((char *)a1 + (v4 & 0xFFFFFFFFFFFFFFC0LL));
  v32 = __ROL8__(v5 ^ 0xB492B66FBE98F273LL, 15);
  v33 = 0xB492B66FBE98F273LL * v5;
  v31 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v5 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v34 = (v5 >> 47) ^ v5;
  v9 = 0x9DDFEA08EB382D69LL
     * ((0x9DDFEA08EB382D69LL * (v34 ^ (0xB492B66FBE98F273LL * v5)))
      ^ v34
      ^ ((0x9DDFEA08EB382D69LL * (v34 ^ (0xB492B66FBE98F273LL * v5))) >> 47));
  v35 = 0x9DDFEA08EB382D69LL * ((v9 >> 47) ^ v9);
  sub_1593A20(&v29, a1);
  v10 = a1 + 8;
  if ( v8 != a1 + 8 )
  {
    v11 = v29;
    v12 = v30;
    v13 = v32;
    v14 = v33;
    v15 = v35;
    v16 = v31;
    v17 = v34;
    do
    {
      v18 = v16;
      v19 = v12 + v13;
      v20 = __ROL8__(v10[6] + v14 + v12, 22);
      v21 = *v10 - 0x4B6D499041670D8DLL * v14;
      v22 = __ROL8__(v11 + v10[1] + v19, 27);
      v12 = v10[5] + v13 - 0x4B6D499041670D8DLL * v20;
      v11 = 0xB492B66FBE98F273LL * __ROL8__(v18 + v17, 31);
      v23 = v21 + v10[2] + v10[1];
      v16 = v15 ^ (0xB492B66FBE98F273LL * v22);
      v13 = v10[3] + v23;
      v14 = __ROR8__(v16 + v21 + v10[3] + v17, 21) + __ROL8__(v23, 20) + v21;
      v24 = v15 + v10[4] + v11;
      v25 = __ROR8__(v24 + v12 + v10[2] + v10[7], 21);
      v26 = __ROL8__(v24 + v10[5] + v10[6], 20);
      v17 = v10[7] + v24 + v10[5] + v10[6];
      v10 += 8;
      v15 = v25 + v26 + v24;
    }
    while ( v8 != v10 );
    v29 = v11;
    v30 = v12;
    v32 = v13;
    v33 = v14;
    v35 = v15;
    v31 = v16;
    v34 = v17;
  }
  if ( (v4 & 0x3F) != 0 )
    sub_1593A20(&v29, (_QWORD *)(a2 - 64));
  v27 = v29
      + 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v35 ^ v33)) ^ v35 ^ ((0x9DDFEA08EB382D69LL * (v35 ^ v33)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v35 ^ v33)) ^ v35 ^ ((0x9DDFEA08EB382D69LL * (v35 ^ v33)) >> 47))))
      - 0x4B6D499041670D8DLL * ((v4 >> 47) ^ v4);
  v28 = 0x9DDFEA08EB382D69LL
      * (v27
       ^ (v31
        - 0x4B6D499041670D8DLL * (v30 ^ (v30 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34 ^ ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34 ^ ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v28 ^ v27 ^ (v28 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v28 ^ v27 ^ (v28 >> 47))));
}
