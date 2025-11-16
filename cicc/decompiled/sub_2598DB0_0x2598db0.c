// Function: sub_2598DB0
// Address: 0x2598db0
//
__int64 __fastcall sub_2598DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __m128i v6; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int8 *v12; // r13
  unsigned __int8 *v13; // r12
  unsigned int v14; // r14d
  int v15; // edx
  int v16; // eax
  __int64 v17; // r13
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rbx
  int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rcx
  int v28; // edx
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r13
  char v34; // al
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  __m128i v37; // rax
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-120h]
  __int64 v44; // [rsp+10h] [rbp-118h]
  char v47; // [rsp+36h] [rbp-F2h] BYREF
  bool v48; // [rsp+37h] [rbp-F1h] BYREF
  __int64 v49; // [rsp+38h] [rbp-F0h] BYREF
  __int64 v50; // [rsp+40h] [rbp-E8h] BYREF
  __m128i v51; // [rsp+48h] [rbp-E0h] BYREF
  _QWORD v52[2]; // [rsp+58h] [rbp-D0h] BYREF
  __int64 v53[2]; // [rsp+68h] [rbp-C0h] BYREF
  _QWORD v54[4]; // [rsp+78h] [rbp-B0h] BYREF
  _QWORD v55[6]; // [rsp+98h] [rbp-90h] BYREF
  _QWORD v56[12]; // [rsp+C8h] [rbp-60h] BYREF

  v2 = a1;
  v3 = (__int64 *)(a1 + 72);
  v4 = sub_25294B0(a2, *(_QWORD *)(v2 + 72), *(_QWORD *)(v2 + 80), v2, 2, 0, 1);
  v44 = v4;
  if ( !v4 )
  {
    v36 = sub_250D070((_QWORD *)(a1 + 72));
    v37.m128i_i64[0] = sub_250D2C0(v36, 0);
    v51 = v37;
    sub_2596DB0(a2, a1, &v51, 0, &v47, 0, 0);
    goto LABEL_4;
  }
  if ( (*(_BYTE *)(v4 + 97) & 3) == 3 )
  {
    sub_250ED80(a2, v4, a1, 1);
    return 1;
  }
  v5 = sub_250D070((_QWORD *)(a1 + 72));
  v6.m128i_i64[0] = sub_250D2C0(v5, 0);
  v51 = v6;
  if ( !(unsigned __int8)sub_2596DB0(a2, a1, &v51, 0, &v47, 0, 0) )
  {
LABEL_4:
    *(_BYTE *)(v2 + 97) = *(_BYTE *)(v2 + 96);
    return 0;
  }
  v52[1] = a1;
  v52[0] = a2;
  v8 = sub_250D070((_QWORD *)(a1 + 72));
  v9 = sub_250D2C0(v8, 0);
  v53[1] = v10;
  v53[0] = v9;
  v49 = sub_25096F0(v53);
  v54[1] = &v49;
  v54[3] = v52;
  v54[0] = a1;
  v54[2] = a2;
  v50 = 0;
  if ( (unsigned __int8)sub_25890A0(a2, a1, v53, 2, &v48, 0, &v50) )
  {
LABEL_6:
    v11 = v50;
    if ( !v50 )
      goto LABEL_7;
    goto LABEL_42;
  }
  v11 = v50;
  if ( !v50 || (*(_WORD *)(v50 + 98) & 3) != 3 )
  {
    v18 = sub_250D070((_QWORD *)(a1 + 72));
    if ( !(unsigned __int8)sub_252FFB0(
                             a2,
                             (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2588100,
                             (__int64)v54,
                             a1,
                             v18,
                             0,
                             1,
                             1,
                             0,
                             0) )
      goto LABEL_4;
    goto LABEL_6;
  }
LABEL_42:
  sub_250ED80(a2, v11, a1, 1);
LABEL_7:
  v12 = (unsigned __int8 *)(*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v12 = (unsigned __int8 *)*((_QWORD *)v12 + 3);
  v13 = v12;
  v14 = 0;
  v43 = 0;
  v15 = *v12;
  v16 = v15 - 29;
  if ( v15 != 40 )
    goto LABEL_10;
LABEL_30:
  v17 = 32LL * (unsigned int)sub_B491D0((__int64)v13);
  if ( (v13[7] & 0x80u) == 0 )
    goto LABEL_31;
  while ( 1 )
  {
    v19 = sub_BD2BC0((__int64)v13);
    v21 = v19 + v20;
    if ( (v13[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v21 >> 4) )
LABEL_54:
        BUG();
LABEL_31:
      v25 = 0;
      goto LABEL_24;
    }
    if ( !(unsigned int)((v21 - sub_BD2BC0((__int64)v13)) >> 4) )
      goto LABEL_31;
    if ( (v13[7] & 0x80u) == 0 )
      goto LABEL_54;
    v22 = *(_DWORD *)(sub_BD2BC0((__int64)v13) + 8);
    if ( (v13[7] & 0x80u) == 0 )
      BUG();
    v23 = sub_BD2BC0((__int64)v13);
    v25 = 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
LABEL_24:
    if ( v14 >= (unsigned int)((32LL * (*((_DWORD *)v13 + 1) & 0x7FFFFFF) - 32 - v17 - v25) >> 5) )
      return 1;
    if ( v14 == (unsigned int)sub_250CB50(v3, 1) )
      goto LABEL_29;
    v26 = *(_QWORD *)&v13[32 * (v14 - (unsigned __int64)(*((_DWORD *)v13 + 1) & 0x7FFFFFF))];
    v27 = *(_QWORD *)(v26 + 8);
    v28 = *(unsigned __int8 *)(v27 + 8);
    if ( (unsigned int)(v28 - 17) <= 1 )
      LOBYTE(v28) = *(_BYTE *)(**(_QWORD **)(v27 + 16) + 8LL);
    if ( (_BYTE)v28 != 14 )
      goto LABEL_29;
    v30 = sub_254C9B0((__int64)v13, v14);
    v32 = sub_25294B0(a2, v30, v31, a1, 2, 0, 1);
    v33 = v32;
    if ( v32 )
    {
      v34 = *(_BYTE *)(v32 + 97);
      if ( (v34 & 3) == 3 )
      {
        sub_250ED80(a2, v33, a1, 1);
        goto LABEL_29;
      }
      if ( (*(_BYTE *)(v44 + 97) & 2) != 0 && (v34 & 2) != 0 )
      {
        sub_250ED80(a2, v44, a1, 1);
        sub_250ED80(a2, v33, a1, 1);
        goto LABEL_29;
      }
    }
    if ( !v43 )
    {
      v38 = *(_QWORD *)(a2 + 208);
      v39 = sub_25096F0(v3);
      v40 = *(_QWORD *)(v38 + 240);
      v41 = *(_QWORD *)v40;
      if ( !*(_QWORD *)v40 )
        goto LABEL_38;
      if ( *(_BYTE *)(v40 + 16) )
      {
        v42 = sub_BBB550(v41, (__int64)&unk_4F86540, v39);
        if ( !v42 )
        {
LABEL_38:
          v2 = a1;
          goto LABEL_4;
        }
      }
      else
      {
        v42 = sub_BC1CD0(v41, &unk_4F86540, v39);
      }
      v43 = v42 + 8;
    }
    v35 = sub_250D070(v3);
    v56[0] = v26;
    v56[1] = -1;
    memset(&v56[2], 0, 32);
    v55[0] = v35;
    v55[1] = -1;
    memset(&v55[2], 0, 32);
    if ( (unsigned __int8)sub_CF4E00(v43, (__int64)v55, (__int64)v56) )
      goto LABEL_38;
LABEL_29:
    v29 = *v13;
    ++v14;
    v16 = v29 - 29;
    if ( v29 == 40 )
      goto LABEL_30;
LABEL_10:
    v17 = 0;
    if ( v16 != 56 )
    {
      if ( v16 != 5 )
        BUG();
      v17 = 64;
    }
    if ( (v13[7] & 0x80u) == 0 )
      goto LABEL_31;
  }
}
