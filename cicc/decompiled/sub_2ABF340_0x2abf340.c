// Function: sub_2ABF340
// Address: 0x2abf340
//
unsigned __int64 __fastcall sub_2ABF340(__int64 *a1, __int64 *a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // rsi
  __int64 v5; // rax
  unsigned __int64 v6; // rsi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r9
  char *v10; // r8
  __int64 v11; // rdx
  char *v12; // rsi
  char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r10
  __int64 v19; // r11
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // r10
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // rcx
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rax
  char *src; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v32; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v33)(__int64, __int64); // [rsp+18h] [rbp-A8h]
  unsigned __int64 v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  unsigned __int64 v36; // [rsp+30h] [rbp-90h]
  unsigned __int64 v37; // [rsp+38h] [rbp-88h]
  unsigned __int64 v38; // [rsp+40h] [rbp-80h]
  __int64 i; // [rsp+50h] [rbp-70h] BYREF
  __int64 v40; // [rsp+58h] [rbp-68h] BYREF
  __int64 v41; // [rsp+60h] [rbp-60h]
  __int64 v42; // [rsp+68h] [rbp-58h]
  __int64 v43; // [rsp+70h] [rbp-50h]
  __int64 v44; // [rsp+78h] [rbp-48h]
  __int64 v45; // [rsp+80h] [rbp-40h]
  __int64 v46; // [rsp+88h] [rbp-38h]
  char v47[8]; // [rsp+90h] [rbp-30h] BYREF
  _BYTE v48[40]; // [rsp+98h] [rbp-28h] BYREF

  if ( a2 == a1 )
  {
    v6 = 0;
    return sub_AC25F0(&i, v6, (__int64)sub_C64CA0);
  }
  v3 = a1 + 4;
  v4 = &v40;
  i = *a1;
  if ( a1 + 4 == a2 )
  {
LABEL_5:
    v6 = (char *)v4 - (char *)&i;
    return sub_AC25F0(&i, v6, (__int64)sub_C64CA0);
  }
  while ( ++v4 != (__int64 *)v48 )
  {
    v5 = *v3;
    v3 += 4;
    *(v4 - 1) = v5;
    if ( v3 == a2 )
      goto LABEL_5;
  }
  v33 = sub_C64CA0;
  v8 = 64;
  v32 = 0;
  v35 = __ROL8__((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL, 15);
  v36 = 0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0;
  v34 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
          ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
          ^ 0xB492B66FBE98F273LL)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 ^ 0xB492B66FBE98F273LL))
         ^ 0xB492B66FBE98F273LL)));
  v37 = (unsigned __int64)sub_C64CA0 ^ ((unsigned __int64)sub_C64CA0 >> 47);
  v9 = 0x9DDFEA08EB382D69LL
     * ((0x9DDFEA08EB382D69LL * (v37 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0)))
      ^ v37
      ^ ((0x9DDFEA08EB382D69LL * (v37 ^ (0xB492B66FBE98F273LL * (_QWORD)sub_C64CA0))) >> 47));
  v38 = 0x9DDFEA08EB382D69LL * ((v9 >> 47) ^ v9);
  sub_AC2A10(&v32, &i);
  v10 = (char *)&i;
  do
  {
    if ( v3 == a2 )
    {
      v12 = v10;
    }
    else
    {
      v11 = *v3;
      v12 = (char *)&v40;
      v3 += 4;
      for ( i = v11; a2 != v3; *((_QWORD *)v13 - 1) = v14 )
      {
        v13 = v12 + 8;
        if ( v12 + 8 == v48 )
          break;
        v14 = *v3;
        v12 += 8;
        v3 += 4;
      }
      v8 += v12 - v10;
    }
    src = v10;
    sub_2AA8DD0(v10, v12, v47);
    v15 = i - 0x4B6D499041670D8DLL * v36;
    v16 = v44 + v35 - 0x4B6D499041670D8DLL * __ROL8__((char *)v33 + v36 + v45, 22);
    v17 = v38 ^ (0xB492B66FBE98F273LL * __ROL8__((char *)v33 + v32 + v35 + v40, 27));
    v18 = v15 + v41 + v40;
    v19 = v42 + v18;
    v33 = (void (__fastcall *)(__int64, __int64))v16;
    v35 = v42 + v18;
    v20 = 0xB492B66FBE98F273LL * __ROL8__(v37 + v34, 31);
    v34 = v17;
    v21 = __ROR8__(v17 + v15 + v42 + v37, 21) + __ROL8__(v18, 20) + v15;
    v22 = v20 + v43 + v38;
    v36 = v21;
    v32 = v20;
    v23 = v22 + v44 + v45;
    v24 = v46 + v23;
    v25 = __ROR8__(v16 + v22 + v41 + v46, 21);
    v37 = v46 + v23;
    v26 = __ROL8__(v23, 20) + v22;
    v10 = src;
    v27 = v25 + v26;
    v38 = v27;
  }
  while ( a2 != v3 );
  v28 = 0xB492B66FBE98F273LL * ((v8 >> 47) ^ v8)
      + v20
      - 0x622015F714C7D297LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v27 ^ v21)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v27 ^ v21)) ^ v27)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v27 ^ v21)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v27 ^ v21)) ^ v27)));
  v29 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v24 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v24 ^ v19)) ^ v24)) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v24 ^ v19)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v24 ^ v19)) ^ v24)));
  v30 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v28 ^ (0xB492B66FBE98F273LL * ((v16 >> 47) ^ v16) + v17 + v29))) >> 47)
       ^ v28
       ^ (0x9DDFEA08EB382D69LL * (v28 ^ (0xB492B66FBE98F273LL * ((v16 >> 47) ^ v16) + v17 + v29))));
  return 0x9DDFEA08EB382D69LL * ((v30 >> 47) ^ v30);
}
