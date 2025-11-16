// Function: sub_CA5190
// Address: 0xca5190
//
_QWORD *__fastcall sub_CA5190(unsigned __int64 *dest, _QWORD *a2, _QWORD *a3, unsigned __int64 a4, __int64 a5)
{
  _QWORD *v6; // r8
  unsigned __int64 v9; // r14
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned __int64 v13; // r11
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r8
  __int64 v21; // rcx
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // r10
  __int64 v25; // rdi
  unsigned __int64 v26; // r11
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r10
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rsi
  _QWORD src[7]; // [rsp+8h] [rbp-38h] BYREF

  src[0] = a5;
  v6 = a3 + 1;
  if ( a4 >= (unsigned __int64)(a3 + 1) )
  {
    *a3 = src[0];
  }
  else
  {
    v9 = a4 - (_QWORD)a3;
    memcpy(a3, src, a4 - (_QWORD)a3);
    if ( *a2 )
    {
      sub_AC2A10(dest + 8, dest);
      *a2 += 64LL;
    }
    else
    {
      v11 = dest[15];
      v12 = __ROL8__(v11 ^ 0xB492B66FBE98F273LL, 15);
      v13 = 0xB492B66FBE98F273LL * v11;
      v14 = v11 ^ (v11 >> 47);
      v15 = __ROL8__(v12 + v11 + dest[1], 27);
      v16 = 0xB492B66FBE98F273LL * __ROL8__(0xB492B66FBE98F273LL * v11 + v11 + dest[6], 22) + dest[5] + v12;
      dest[9] = v16;
      v17 = (0x9DDFEA08EB382D69LL
           * (((0x9DDFEA08EB382D69LL
              * (v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)) ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47))) >> 47)
            ^ (0x9DDFEA08EB382D69LL
             * (v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)) ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47)))))
          ^ (0xB492B66FBE98F273LL * v15);
      v18 = 0x9DDFEA08EB382D69LL
          * (((0x9DDFEA08EB382D69LL
             * (((0x9DDFEA08EB382D69LL * (v11 ^ 0xB492B66FBE98F273LL)) >> 47)
              ^ (0x9DDFEA08EB382D69LL * (v11 ^ 0xB492B66FBE98F273LL))
              ^ 0xB492B66FBE98F273LL)) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL * (v11 ^ 0xB492B66FBE98F273LL)) >> 47)
             ^ (0x9DDFEA08EB382D69LL * (v11 ^ 0xB492B66FBE98F273LL))
             ^ 0xB492B66FBE98F273LL)));
      v19 = dest[4]
          - 0x622015F714C7D297LL
          * (((0x9DDFEA08EB382D69LL
             * (v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)) ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47))) >> 47)
           ^ (0x9DDFEA08EB382D69LL
            * (v14 ^ (0x9DDFEA08EB382D69LL * (v14 ^ v13)) ^ ((0x9DDFEA08EB382D69LL * (v14 ^ v13)) >> 47))));
      v20 = dest[1];
      v21 = *dest - 0x6D8ED9027DD26057LL * v11;
      dest[10] = v17;
      v22 = v21 + dest[2] + v20;
      v23 = 0xB492B66FBE98F273LL * __ROL8__(v14 + v18, 31);
      v24 = dest[5] + dest[6];
      v25 = __ROR8__(dest[3] + v21 + v14 + v17, 21);
      v26 = v19 + v23;
      v27 = dest[3];
      dest[8] = v23;
      v28 = v26 + v24;
      dest[11] = v22 + v27;
      v29 = v28 + dest[7];
      v30 = dest[2] + dest[7] + v16;
      dest[12] = __ROL8__(v22, 20) + v21 + v25;
      dest[13] = v29;
      dest[14] = v26 + __ROL8__(v28, 20) + __ROR8__(v30 + v26, 21);
      *a2 = 64;
    }
    if ( a4 < (unsigned __int64)dest + 8 - v9 )
      BUG();
    memcpy(dest, (char *)src + v9, 8 - v9);
    return (unsigned __int64 *)((char *)dest + 8 - v9);
  }
  return v6;
}
