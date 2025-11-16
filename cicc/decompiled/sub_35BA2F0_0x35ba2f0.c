// Function: sub_35BA2F0
// Address: 0x35ba2f0
//
unsigned __int64 __fastcall sub_35BA2F0(unsigned int *a1, unsigned int *a2)
{
  __int64 v2; // rax
  unsigned int *v3; // rbx
  __int64 *v5; // rdi
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  char *v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // r10
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // r11
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r11
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  char *v23; // rdx
  char *v24; // r13
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // rsi
  unsigned __int64 v30; // [rsp+8h] [rbp-D8h]
  char *v31; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v32; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v33; // [rsp+30h] [rbp-B0h] BYREF
  void (__fastcall *v34)(__int64, __int64); // [rsp+38h] [rbp-A8h]
  unsigned __int64 v35; // [rsp+40h] [rbp-A0h]
  __int64 v36; // [rsp+48h] [rbp-98h]
  unsigned __int64 v37; // [rsp+50h] [rbp-90h]
  unsigned __int64 v38; // [rsp+58h] [rbp-88h]
  unsigned __int64 v39; // [rsp+60h] [rbp-80h]
  __int64 v40; // [rsp+70h] [rbp-70h] BYREF
  __int64 v41; // [rsp+78h] [rbp-68h] BYREF
  __int64 v42; // [rsp+80h] [rbp-60h]
  __int64 v43; // [rsp+88h] [rbp-58h]
  __int64 v44; // [rsp+90h] [rbp-50h]
  __int64 v45; // [rsp+98h] [rbp-48h]
  __int64 v46; // [rsp+A0h] [rbp-40h]
  __int64 v47; // [rsp+A8h] [rbp-38h]
  char v48[8]; // [rsp+B0h] [rbp-30h] BYREF
  _BYTE v49[40]; // [rsp+B8h] [rbp-28h] BYREF

  if ( a1 == a2 )
    return sub_AC25F0(&v40, 0, (__int64)sub_C64CA0);
  v2 = *a1;
  v3 = a1;
  v5 = &v41;
  v6 = 0x9DDFEA08EB382D69LL
     * ((0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * v2))
      ^ ((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * v2)) >> 47));
  v7 = 0x9DDFEA08EB382D69LL * (v6 ^ (v6 >> 47));
  do
  {
    ++v3;
    *(v5 - 1) = v7;
    if ( a2 == v3 )
      return sub_AC25F0(&v40, (char *)v5 - (char *)&v40, (__int64)sub_C64CA0);
    ++v5;
    v8 = 0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *v3);
    v7 = 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v8 ^ (v8 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v8 ^ (v8 >> 47))));
  }
  while ( v5 != (__int64 *)v49 );
  v34 = sub_C64CA0;
  v33 = 0;
  v36 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v38 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v37 = 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0;
  v35 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v39 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * (v38 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
          ^ v38
          ^ ((0x9DDFEA08EB382D69LL * (v38 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * (v38 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
         ^ v38
         ^ ((0x9DDFEA08EB382D69LL * (v38 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47))));
  sub_AC2A10(&v33, &v40);
  v9 = v49;
  v32 = 64;
  do
  {
    v23 = (char *)&v40;
    while ( 1 )
    {
      v24 = v23;
      v23 += 8;
      v25 = 0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *v3);
      if ( v23 == v9 )
        break;
      ++v3;
      *((_QWORD *)v23 - 1) = 0x9DDFEA08EB382D69LL
                           * (((0x9DDFEA08EB382D69LL * (v25 ^ (v25 >> 47))) >> 47)
                            ^ (0x9DDFEA08EB382D69LL * (v25 ^ (v25 >> 47))));
      if ( a2 == v3 )
      {
        v24 = v23;
        break;
      }
    }
    v31 = v9;
    sub_35B8550((char *)&v40, v24, v48);
    v10 = v40 - 0x4B6D499041670D8DLL * v37;
    v11 = 0xB492B66FBE98F273LL * __ROL8__((char *)v34 + v37 + v46, 22) + v45 + v36;
    v30 = v39 ^ (0xB492B66FBE98F273LL * __ROL8__((char *)v34 + v33 + v36 + v41, 27));
    v12 = v38 + v35;
    v35 = v30;
    v13 = v10 + v42 + v41;
    v14 = v43 + v13;
    v15 = 0xB492B66FBE98F273LL * __ROL8__(v12, 31);
    v34 = (void (__fastcall *)(__int64, __int64))v11;
    v16 = v30 + v10 + v43 + v38;
    v17 = __ROL8__(v13, 20) + v10;
    v36 = v43 + v13;
    v18 = v44 + v39 + v15;
    v19 = v17 + __ROR8__(v16, 21);
    v37 = v19;
    v33 = v15;
    v20 = v18 + v45 + v46;
    v38 = v47 + v20;
    v22 = __ROR8__(v11 + v18 + v42 + v47, 21) + __ROL8__(v20, 20) + v18;
    v39 = v22;
    v32 += v24 - (char *)&v40;
    v9 = v31;
  }
  while ( a2 != v3 );
  v26 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v22 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v22 ^ v19)) ^ v22)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v22 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v22 ^ v19)) ^ v22)))
      + 0xB492B66FBE98F273LL * (v32 ^ (v32 >> 47))
      + v15;
  v27 = 0xB492B66FBE98F273LL * (v11 ^ (v11 >> 47));
  v21 = v47 + v20;
  v28 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v21 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v21 ^ v14)) ^ v21)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v21 ^ v14)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v21 ^ v14)) ^ v21)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v26 ^ (v27 + v30 + v28))) >> 47)
           ^ v26
           ^ (0x9DDFEA08EB382D69LL * (v26 ^ (v27 + v30 + v28))))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v26 ^ (v27 + v30 + v28))) >> 47)
          ^ v26
          ^ (0x9DDFEA08EB382D69LL * (v26 ^ (v27 + v30 + v28))))));
}
