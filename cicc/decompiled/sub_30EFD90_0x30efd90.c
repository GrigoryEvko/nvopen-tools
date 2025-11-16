// Function: sub_30EFD90
// Address: 0x30efd90
//
__int64 __fastcall sub_30EFD90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v7; // r12
  unsigned __int8 v8; // bl
  _QWORD *v9; // rax
  char v11; // dl
  __int64 v12; // rcx
  unsigned __int8 *v13; // r13
  unsigned __int8 v14; // al
  int v15; // edi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rdx
  __int64 *v25; // rax
  char v26; // al
  __int64 v27; // r12
  __int64 *v28; // rax
  char v29; // dl
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+18h] [rbp-338h]
  unsigned __int8 *v34; // [rsp+20h] [rbp-330h] BYREF
  __int16 v35; // [rsp+28h] [rbp-328h]
  __int64 v36; // [rsp+30h] [rbp-320h] BYREF
  __int64 *v37; // [rsp+38h] [rbp-318h]
  __int64 v38; // [rsp+40h] [rbp-310h]
  int v39; // [rsp+48h] [rbp-308h]
  char v40; // [rsp+4Ch] [rbp-304h]
  char v41; // [rsp+50h] [rbp-300h] BYREF
  __m128i v42; // [rsp+70h] [rbp-2E0h] BYREF
  __int64 v43; // [rsp+80h] [rbp-2D0h]
  __int64 v44; // [rsp+88h] [rbp-2C8h]
  __int64 v45; // [rsp+90h] [rbp-2C0h] BYREF
  __int64 v46; // [rsp+98h] [rbp-2B8h]
  __int64 v47; // [rsp+A0h] [rbp-2B0h]
  __int64 v48; // [rsp+A8h] [rbp-2A8h]
  __int16 v49; // [rsp+B0h] [rbp-2A0h]
  _QWORD v50[2]; // [rsp+1D0h] [rbp-180h] BYREF
  char v51; // [rsp+1E0h] [rbp-170h]
  _BYTE *v52; // [rsp+1E8h] [rbp-168h]
  __int64 v53; // [rsp+1F0h] [rbp-160h]
  _BYTE v54[128]; // [rsp+1F8h] [rbp-158h] BYREF
  __int16 v55; // [rsp+278h] [rbp-D8h]
  _QWORD v56[2]; // [rsp+280h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+290h] [rbp-C0h]
  __int64 v58; // [rsp+298h] [rbp-B8h] BYREF
  unsigned int v59; // [rsp+2A0h] [rbp-B0h]
  _BYTE v60[56]; // [rsp+318h] [rbp-38h] BYREF

  while ( 1 )
  {
    v6 = a1;
    v7 = a4;
    v8 = a3;
    if ( !*(_BYTE *)(a4 + 28) )
      goto LABEL_8;
    v9 = *(_QWORD **)(a4 + 8);
    a4 = *(unsigned int *)(a4 + 20);
    a3 = (__int64)&v9[a4];
    if ( v9 != (_QWORD *)a3 )
    {
      while ( a2 != *v9 )
      {
        if ( (_QWORD *)a3 == ++v9 )
          goto LABEL_7;
      }
      return sub_ACADE0(*(__int64 ***)(a2 + 8));
    }
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v7 + 16) )
    {
      *(_DWORD *)(v7 + 20) = a4 + 1;
      *(_QWORD *)a3 = a2;
      ++*(_QWORD *)v7;
      if ( !v8 )
        goto LABEL_10;
    }
    else
    {
LABEL_8:
      sub_C8CC70(v7, a2, a3, a4, a5, a6);
      if ( !v11 )
        return sub_ACADE0(*(__int64 ***)(a2 + 8));
      if ( !v8 )
      {
LABEL_10:
        v13 = sub_BD3990((unsigned __int8 *)a2, a2);
        v14 = *v13;
        if ( *v13 <= 0x1Cu )
          goto LABEL_11;
        goto LABEL_16;
      }
    }
    v13 = sub_98ACB0((unsigned __int8 *)a2, 6u);
    v14 = *v13;
    if ( *v13 <= 0x1Cu )
    {
LABEL_11:
      if ( v14 != 5 )
        goto LABEL_30;
      v15 = *((unsigned __int16 *)v13 + 1);
      if ( (unsigned int)(v15 - 38) > 0xC )
      {
LABEL_32:
        a2 = sub_97B670(v13, v6[2], v6[6]);
        if ( v13 == (unsigned __int8 *)a2 )
          return (__int64)v13;
        goto LABEL_21;
      }
      if ( !sub_B50750(
              v15,
              *(_QWORD *)(*(_QWORD *)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)] + 8LL),
              *((_QWORD *)v13 + 1),
              v6[2]) )
        goto LABEL_29;
      a3 = v8;
      a2 = *(_QWORD *)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)];
      goto LABEL_22;
    }
LABEL_16:
    if ( v14 == 61 )
      break;
    if ( v14 == 84 )
    {
      a2 = sub_B48DC0((__int64)v13);
      if ( !a2 )
        goto LABEL_29;
      goto LABEL_21;
    }
    if ( (unsigned int)v14 - 67 <= 0xC )
    {
      if ( sub_B507F0(v13, a1[2]) )
      {
        a2 = *((_QWORD *)v13 - 4);
        a3 = v8;
        goto LABEL_22;
      }
LABEL_29:
      v14 = *v13;
      if ( *v13 <= 0x1Cu )
      {
LABEL_30:
        if ( v14 > 0x15u )
          return (__int64)v13;
        goto LABEL_32;
      }
LABEL_20:
      v16 = v6[2];
      v17 = v6[4];
      v43 = 0;
      v18 = v6[6];
      v19 = v6[5];
      v46 = 0;
      v42.m128i_i64[0] = v16;
      v45 = v17;
      v42.m128i_i64[1] = v18;
      v44 = v19;
      v47 = 0;
      v48 = 0;
      v49 = 257;
      a2 = sub_1020E10((__int64)v13, &v42, v19, v18, a5, a6);
      if ( !a2 )
        return (__int64)v13;
LABEL_21:
      a3 = v8;
LABEL_22:
      a4 = v7;
      goto LABEL_23;
    }
    if ( v14 != 93 )
      goto LABEL_20;
    LOBYTE(v43) = 0;
    v32 = sub_98A4C0(
            *((_QWORD *)v13 - 4),
            *((unsigned int **)v13 + 9),
            *((unsigned int *)v13 + 20),
            v12,
            a5,
            a6,
            v42.m128i_i64[0],
            v42.m128i_i64[1],
            v43);
    if ( !v32 || v13 == (unsigned __int8 *)v32 )
      goto LABEL_29;
    a3 = v8;
    a4 = v7;
    a2 = v32;
LABEL_23:
    a1 = v6;
  }
  v20 = 0;
  v21 = *((_QWORD *)v13 + 5);
  v36 = 0;
  v34 = v13 + 24;
  v37 = (__int64 *)&v41;
  v22 = a1[3];
  v35 = 0;
  v38 = 4;
  v39 = 0;
  v40 = 1;
  v43 = 0;
  v44 = 1;
  v42.m128i_i64[0] = v22;
  v42.m128i_i64[1] = v22;
  v23 = &v45;
  do
  {
    *v23 = -4;
    v23 += 5;
    *(v23 - 4) = -3;
    *(v23 - 3) = -4;
    *(v23 - 2) = -3;
  }
  while ( v23 != v50 );
  v24 = (__int64 *)v60;
  v50[1] = 0;
  v50[0] = v56;
  v52 = v54;
  v53 = 0x400000000LL;
  v51 = 0;
  v55 = 256;
  v56[1] = 0;
  v57 = 1;
  v56[0] = &unk_49DDBE8;
  v25 = &v58;
  do
  {
    *v25 = -4096;
    v25 += 2;
  }
  while ( v25 != (__int64 *)v60 );
  v33 = v7;
  v26 = 1;
  v27 = v21;
  while ( 1 )
  {
    if ( !v26 )
      goto LABEL_52;
    v28 = v37;
    v12 = HIDWORD(v38);
    v24 = &v37[HIDWORD(v38)];
    if ( v37 != v24 )
    {
      while ( v27 != *v28 )
      {
        if ( v24 == ++v28 )
          goto LABEL_57;
      }
LABEL_44:
      v7 = v33;
      v56[0] = &unk_49DDBE8;
      if ( (v57 & 1) == 0 )
        sub_C7D6A0(v58, 16LL * v59, 8);
      nullsub_184();
      if ( v52 != v54 )
        _libc_free((unsigned __int64)v52);
      if ( (v44 & 1) == 0 )
        sub_C7D6A0(v45, 40LL * (unsigned int)v46, 8);
      if ( !v40 )
        _libc_free((unsigned __int64)v37);
      goto LABEL_29;
    }
LABEL_57:
    if ( HIDWORD(v38) < (unsigned int)v38 )
    {
      ++HIDWORD(v38);
      *v24 = v27;
      ++v36;
    }
    else
    {
LABEL_52:
      sub_C8CC70((__int64)&v36, v27, (__int64)v24, v12, v20, a6);
      if ( !v29 )
        goto LABEL_44;
    }
    v30 = sub_D319E0((__int64)v13, v27, &v34, qword_4F86CA8[8], &v42, 0, 0);
    if ( v30 )
      break;
    if ( v34 != *(unsigned __int8 **)(v27 + 56) )
      goto LABEL_44;
    v31 = sub_AA5510(v27);
    v27 = v31;
    if ( !v31 )
      goto LABEL_44;
    v24 = 0;
    v34 = (unsigned __int8 *)(v31 + 48);
    v26 = v40;
    v35 = 0;
  }
  v13 = (unsigned __int8 *)sub_30EFD90(a1, v30, v8, v33);
  v56[0] = &unk_49DDBE8;
  if ( (v57 & 1) == 0 )
    sub_C7D6A0(v58, 16LL * v59, 8);
  nullsub_184();
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  if ( (v44 & 1) == 0 )
    sub_C7D6A0(v45, 40LL * (unsigned int)v46, 8);
  if ( !v40 )
    _libc_free((unsigned __int64)v37);
  return (__int64)v13;
}
