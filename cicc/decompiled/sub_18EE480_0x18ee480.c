// Function: sub_18EE480
// Address: 0x18ee480
//
__int64 __fastcall sub_18EE480(
        _QWORD *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r15d
  __int64 *v12; // rbx
  unsigned int v13; // eax
  int v14; // ebx
  __int64 v15; // r13
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r13
  const char *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 *v24; // r15
  const char *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r13
  const char *v29; // rax
  int v30; // edi
  __int64 v31; // rdx
  __int64 v32; // r13
  const char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r14
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v40; // rbx
  unsigned __int64 v41; // rbx
  bool v42; // al
  unsigned int v43; // [rsp+Ch] [rbp-A4h]
  __int64 *v44; // [rsp+18h] [rbp-98h]
  unsigned __int64 v45; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v46; // [rsp+28h] [rbp-88h]
  __int64 v47; // [rsp+30h] [rbp-80h]
  unsigned int v48; // [rsp+38h] [rbp-78h]
  const char *v49; // [rsp+40h] [rbp-70h] BYREF
  __int64 v50; // [rsp+48h] [rbp-68h]
  __int64 v51; // [rsp+50h] [rbp-60h]
  unsigned int v52; // [rsp+58h] [rbp-58h]
  const char **v53; // [rsp+60h] [rbp-50h] BYREF
  char *v54; // [rsp+68h] [rbp-48h]
  __int64 v55; // [rsp+70h] [rbp-40h]
  unsigned int v56; // [rsp+78h] [rbp-38h]

  v10 = 0;
  if ( *(_BYTE *)(*a1 + 8LL) != 16 )
  {
    v43 = *(_DWORD *)(*a1 + 8LL) >> 8;
    sub_15897D0((__int64)&v45, v43, 0);
    if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
    {
      v12 = (__int64 *)*(a1 - 1);
      v44 = &v12[3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
    }
    else
    {
      v44 = a1;
      v12 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
    }
    for ( ; v44 != v12; v12 += 3 )
    {
      sub_13F2950((__int64)&v49, a2, *v12, a1[5], 0);
      sub_158C3A0((__int64)&v53, (__int64)&v45, (__int64)&v49);
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      v45 = (unsigned __int64)v53;
      v13 = (unsigned int)v54;
      LODWORD(v54) = 0;
      v46 = v13;
      if ( v48 > 0x40 && v47 )
      {
        j_j___libc_free_0_0(v47);
        v47 = v55;
        v48 = v56;
        if ( (unsigned int)v54 > 0x40 && v53 )
          j_j___libc_free_0_0(v53);
      }
      else
      {
        v47 = v55;
        v48 = v56;
      }
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      if ( (unsigned int)v50 > 0x40 && v49 )
        j_j___libc_free_0_0(v49);
    }
    sub_158A9F0((__int64)&v53, (__int64)&v45);
    v14 = (int)v54;
    if ( (unsigned int)v54 > 0x40 )
    {
      v40 = v14 - (unsigned int)sub_16A57B0((__int64)&v53);
      if ( v40 )
      {
        v41 = ((((unsigned __int64)(v40 - 1) >> 1) | (v40 - 1)) >> 2) | ((unsigned __int64)(v40 - 1) >> 1) | (v40 - 1);
        v15 = ((((v41 >> 4) | v41 | (((v41 >> 4) | v41) >> 8)) >> 16) | (v41 >> 4) | v41 | (((v41 >> 4) | v41) >> 8))
            + 1;
        if ( (unsigned int)v15 < 8 )
          LODWORD(v15) = 8;
      }
      else
      {
        LODWORD(v15) = 8;
      }
      if ( v53 )
        j_j___libc_free_0_0(v53);
    }
    else
    {
      LODWORD(v15) = 8;
      if ( v53 )
      {
        _BitScanReverse64(&v16, (unsigned __int64)v53);
        v17 = (((((((64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
                  | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 2)
                | (64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
                | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 4)
              | (((64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
                | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 2)
              | (64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
              | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 8)
            | (((((64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
                | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 2)
              | (64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
              | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 4)
            | (((64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
              | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1)) >> 2)
            | (64 - ((unsigned int)v16 ^ 0x3F) - 1LL)
            | (((unsigned __int64)(64 - ((unsigned int)v16 ^ 0x3F)) - 1) >> 1);
        v18 = ((v17 >> 16) | v17) + 1;
        if ( (unsigned int)v18 >= 8 )
          LODWORD(v15) = v18;
      }
    }
    v10 = 0;
    if ( v43 > (unsigned int)v15 )
    {
      v19 = (_QWORD *)sub_16498A0((__int64)a1);
      v20 = sub_1644C60(v19, v15);
      v21 = sub_1649960((__int64)a1);
      v22 = *(a1 - 6);
      v49 = v21;
      v50 = v23;
      LOWORD(v55) = 773;
      v53 = &v49;
      v54 = ".lhs.trunc";
      v24 = (__int64 *)sub_15FDBD0(36, v22, v20, (__int64)&v53, (__int64)a1);
      v25 = sub_1649960((__int64)a1);
      v26 = *(a1 - 3);
      v50 = v27;
      v49 = v25;
      LOWORD(v55) = 773;
      v53 = &v49;
      v54 = ".rhs.trunc";
      v28 = sub_15FDBD0(36, v26, v20, (__int64)&v53, (__int64)a1);
      v29 = sub_1649960((__int64)a1);
      v30 = *((unsigned __int8 *)a1 + 16);
      v50 = v31;
      LOWORD(v55) = 261;
      v49 = v29;
      v53 = &v49;
      v32 = sub_15FB440(v30 - 24, v24, v28, (__int64)&v53, (__int64)a1);
      v33 = sub_1649960((__int64)a1);
      v50 = v34;
      v35 = *a1;
      v49 = v33;
      LOWORD(v55) = 773;
      v53 = &v49;
      v54 = ".zext";
      v36 = sub_15FDBD0(37, v32, v35, (__int64)&v53, (__int64)a1);
      if ( *(_BYTE *)(v32 + 16) == 41 )
      {
        v42 = sub_15F23D0((__int64)a1);
        sub_15F2350(v32, v42);
      }
      v10 = 1;
      sub_164D160((__int64)a1, v36, a3, a4, a5, a6, v37, v38, a9, a10);
      sub_15F20C0(a1);
    }
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
  }
  return v10;
}
