// Function: sub_1597510
// Address: 0x1597510
//
unsigned __int64 __fastcall sub_1597510(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // r13
  unsigned __int64 v7; // rax
  __int64 *v8; // r12
  __int64 *v9; // rdi
  unsigned __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r10
  __int64 v13; // r8
  __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r11
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v29; // [rsp+18h] [rbp-68h]
  unsigned __int64 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+30h] [rbp-50h]
  unsigned __int64 v33; // [rsp+38h] [rbp-48h]
  __int64 v34; // [rsp+40h] [rbp-40h]

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v7 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v7 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v7;
    sub_2207640(byte_4F99930);
    v4 = qword_4F99938;
    v5 = a2 - (_QWORD)a1;
    if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
      goto LABEL_8;
    return sub_1593600(a1, v5, v4);
  }
  v4 = qword_4F99938;
  v5 = a2 - (_QWORD)a1;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_1593600(a1, v5, v4);
LABEL_8:
  sub_15938B0(&v28, a1, v4);
  v8 = (__int64 *)((char *)a1 + (v5 & 0xFFFFFFFFFFFFFFC0LL));
  v9 = a1 + 8;
  if ( v8 != a1 + 8 )
  {
    v10 = v28;
    v11 = v29;
    v12 = v31;
    v13 = v32;
    v14 = v34;
    v15 = v30;
    v16 = v33;
    do
    {
      v17 = v15;
      v18 = v12 + v11;
      v19 = __ROL8__(v9[6] + v13 + v11, 22);
      v20 = *v9 - 0x4B6D499041670D8DLL * v13;
      v21 = __ROL8__(v10 + v9[1] + v18, 27);
      v11 = v9[5] + v12 - 0x4B6D499041670D8DLL * v19;
      v10 = 0xB492B66FBE98F273LL * __ROL8__(v17 + v16, 31);
      v22 = v20 + v9[2] + v9[1];
      v15 = v14 ^ (0xB492B66FBE98F273LL * v21);
      v12 = v9[3] + v22;
      v13 = __ROR8__(v15 + v20 + v9[3] + v16, 21) + __ROL8__(v22, 20) + v20;
      v23 = v14 + v9[4] + v10;
      v24 = __ROR8__(v23 + v11 + v9[2] + v9[7], 21);
      v25 = __ROL8__(v23 + v9[5] + v9[6], 20);
      v16 = v9[7] + v23 + v9[5] + v9[6];
      v9 += 8;
      v14 = v24 + v25 + v23;
    }
    while ( v8 != v9 );
    v28 = v10;
    v29 = v11;
    v31 = v12;
    v32 = v13;
    v34 = v14;
    v30 = v15;
    v33 = v16;
  }
  if ( (v5 & 0x3F) != 0 )
    sub_1593A20(&v28, (_QWORD *)(a2 - 64));
  v26 = v28
      + 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34 ^ ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34 ^ ((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47))))
      - 0x4B6D499041670D8DLL * ((v5 >> 47) ^ v5);
  v27 = 0x9DDFEA08EB382D69LL
      * (v26
       ^ (v30
        - 0x4B6D499041670D8DLL * (v29 ^ (v29 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * ((0x9DDFEA08EB382D69LL * (v33 ^ v31)) ^ v33 ^ ((0x9DDFEA08EB382D69LL * (v33 ^ v31)) >> 47))) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (v33 ^ v31)) ^ v33 ^ ((0x9DDFEA08EB382D69LL * (v33 ^ v31)) >> 47))))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v27 ^ v26 ^ (v27 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v27 ^ v26 ^ (v27 >> 47))));
}
