// Function: sub_1ECD5B0
// Address: 0x1ecd5b0
//
unsigned __int64 __fastcall sub_1ECD5B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r8
  unsigned __int64 v6; // rax
  _QWORD *v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v11; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-68h]
  unsigned __int64 v13; // [rsp+20h] [rbp-60h]
  __int64 v14; // [rsp+28h] [rbp-58h]
  unsigned __int64 v15; // [rsp+30h] [rbp-50h]
  unsigned __int64 v16; // [rsp+38h] [rbp-48h]
  unsigned __int64 v17; // [rsp+40h] [rbp-40h]

  v2 = a1;
  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v6 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v6 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v6;
    sub_2207640(byte_4F99930);
    v4 = qword_4F99938;
    v3 = a2 - (_QWORD)a1;
    if ( (unsigned __int64)(a2 - (_QWORD)a1) > 0x40 )
      goto LABEL_8;
    return sub_1593600(a1, v3, v4);
  }
  v3 = a2 - (_QWORD)a1;
  v4 = qword_4F99938;
  if ( (unsigned __int64)(a2 - (_QWORD)a1) <= 0x40 )
    return sub_1593600(a1, v3, v4);
LABEL_8:
  v12 = v4;
  v11 = 0;
  v14 = __ROL8__(v4 ^ 0xB492B66FBE98F273LL, 15);
  v15 = 0xB492B66FBE98F273LL * v4;
  v13 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * (v4 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v16 = (v4 >> 47) ^ v4;
  v17 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v16 ^ v15)) >> 47) ^ v16 ^ (0x9DDFEA08EB382D69LL * (v16 ^ v15)))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v16 ^ v15)) >> 47) ^ v16 ^ (0x9DDFEA08EB382D69LL * (v16 ^ v15)))));
  do
  {
    v7 = v2;
    v2 += 8;
    sub_1593A20(&v11, v7);
  }
  while ( (_QWORD *)((char *)a1 + (v3 & 0xFFFFFFFFFFFFFFC0LL)) != v2 );
  if ( (v3 & 0x3F) != 0 )
    sub_1593A20(&v11, (_QWORD *)(a2 - 64));
  v8 = v11
     + 0xB492B66FBE98F273LL * ((v3 >> 47) ^ v3)
     - 0x622015F714C7D297LL
     * (((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v17 ^ v15)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v17 ^ v15)) ^ v17)) >> 47)
      ^ (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v17 ^ v15)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v17 ^ v15)) ^ v17)));
  v9 = 0x9DDFEA08EB382D69LL
     * (v8
      ^ (v13
       - 0x4B6D499041670D8DLL * (v12 ^ (v12 >> 47))
       - 0x622015F714C7D297LL
       * (((0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v16 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v16 ^ v14)) ^ v16)) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v16 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v16 ^ v14)) ^ v16)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * ((v9 >> 47) ^ v9 ^ v8)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v9 >> 47) ^ v9 ^ v8)));
}
