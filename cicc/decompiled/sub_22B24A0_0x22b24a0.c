// Function: sub_22B24A0
// Address: 0x22b24a0
//
unsigned __int64 __fastcall sub_22B24A0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // r8
  _QWORD *v9; // rax
  __int64 v10; // r8
  _QWORD *v11; // rax
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r8
  char *v17; // rax
  __int64 v18; // r14
  signed __int64 v20; // rbx
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v24; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v26; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+38h] [rbp-B8h] BYREF
  _OWORD dest[4]; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v29; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v30; // [rsp+88h] [rbp-68h]
  __int64 v31; // [rsp+90h] [rbp-60h]
  __int64 v32; // [rsp+98h] [rbp-58h]
  __int64 v33; // [rsp+A0h] [rbp-50h]
  __int64 v34; // [rsp+A8h] [rbp-48h]
  __int64 v35; // [rsp+B0h] [rbp-40h]
  void (__fastcall *v36)(__int64, __int64); // [rsp+B8h] [rbp-38h]

  v8 = *a1;
  memset(dest, 0, sizeof(dest));
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = sub_C64CA0;
  v23 = 0;
  v9 = sub_CA5190((unsigned __int64 *)dest, &v23, dest, (unsigned __int64)&v29, v8);
  v10 = *a2;
  v24 = v23;
  v11 = sub_CA5190((unsigned __int64 *)dest, &v24, v9, (unsigned __int64)&v29, v10);
  v12 = *a3;
  v25 = v24;
  v13 = sub_CA5190((unsigned __int64 *)dest, &v25, v11, (unsigned __int64)&v29, v12);
  v14 = *a4;
  v26 = v25;
  v15 = sub_CA5190((unsigned __int64 *)dest, &v26, v13, (unsigned __int64)&v29, v14);
  v16 = *a5;
  v27 = v26;
  v17 = (char *)sub_CA5190((unsigned __int64 *)dest, &v27, v15, (unsigned __int64)&v29, v16);
  v18 = v27;
  if ( !v27 )
    return sub_AC25F0(dest, v17 - (char *)dest, (__int64)v36);
  v20 = v17 - (char *)dest;
  sub_22AD600((char *)dest, v17, (char *)&v29);
  sub_AC2A10(&v29, dest);
  v21 = v29
      - 0x622015F714C7D297LL
      * ((0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (v35 ^ v33)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v35 ^ v33)) ^ v35))
       ^ ((0x9DDFEA08EB382D69LL
         * (((0x9DDFEA08EB382D69LL * (v35 ^ v33)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v35 ^ v33)) ^ v35)) >> 47))
      - 0x4B6D499041670D8DLL * (((unsigned __int64)(v18 + v20) >> 47) ^ (v18 + v20));
  v22 = 0x9DDFEA08EB382D69LL
      * (v21
       ^ (v31
        - 0x4B6D499041670D8DLL * (v30 ^ (v30 >> 47))
        - 0x622015F714C7D297LL
        * (((0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34)) >> 47)
         ^ (0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL * (v34 ^ v32)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v34 ^ v32)) ^ v34)))));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v22 ^ v21 ^ (v22 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v22 ^ v21 ^ (v22 >> 47))));
}
