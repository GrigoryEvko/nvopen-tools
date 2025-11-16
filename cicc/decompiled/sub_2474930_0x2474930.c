// Function: sub_2474930
// Address: 0x2474930
//
void __fastcall sub_2474930(__int64 a1, unsigned __int8 *a2, int a3, unsigned __int8 a4)
{
  int v6; // edx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // eax
  _BYTE *v17; // r12
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // r15
  __int64 (__fastcall *v22)(__int64, _BYTE *, unsigned __int8 *); // rax
  _BYTE *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int8 *v31; // r12
  _BYTE *v32; // r13
  __int64 (__fastcall *v33)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // r9
  __int64 v37; // r15
  unsigned int *v38; // r14
  unsigned int *v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // rsi
  _QWORD *v44; // rax
  unsigned int *v45; // r15
  unsigned int *v46; // r12
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  const __m128i *v52; // rbx
  unsigned __int64 v53; // r8
  __m128i *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdi
  const void *v58; // rsi
  char *v59; // rbx
  __int64 v60; // [rsp+8h] [rbp-168h]
  __int64 v61; // [rsp+18h] [rbp-158h]
  __int64 v64; // [rsp+38h] [rbp-138h]
  _BYTE *v65; // [rsp+40h] [rbp-130h]
  __int64 v66; // [rsp+40h] [rbp-130h]
  _BYTE v67[32]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v68; // [rsp+70h] [rbp-100h]
  _QWORD v69[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v70; // [rsp+A0h] [rbp-D0h]
  unsigned int *v71; // [rsp+B0h] [rbp-C0h] BYREF
  int v72; // [rsp+B8h] [rbp-B8h]
  __int64 v73; // [rsp+E8h] [rbp-88h]
  __int64 v74; // [rsp+F0h] [rbp-80h]
  _QWORD *v75; // [rsp+F8h] [rbp-78h]
  __int64 v76; // [rsp+100h] [rbp-70h]
  __int64 v77; // [rsp+108h] [rbp-68h]

  sub_23D0AB0((__int64)&v71, (__int64)a2, 0, 0, 0);
  v6 = *a2;
  if ( v6 == 40 )
  {
    v7 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v7 = 0;
    if ( v6 != 85 )
    {
      v7 = 64;
      if ( v6 != 34 )
LABEL_61:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v8 = sub_BD2BC0((__int64)a2);
  v10 = v8 + v9;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v10 >> 4) )
LABEL_62:
      BUG();
LABEL_10:
    v14 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_62;
  v11 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v12 = sub_BD2BC0((__int64)a2);
  v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
LABEL_11:
  v15 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v16 = ((32 * v15 - 32 - v7 - v14) >> 5) - a4;
  if ( v16 == 1 )
  {
    v61 = 0;
    v64 = *(_QWORD *)&a2[-32 * v15];
  }
  else
  {
    if ( v16 != 2 )
      goto LABEL_61;
    v61 = *(_QWORD *)&a2[-32 * v15];
    v64 = *(_QWORD *)&a2[32 * (1 - v15)];
  }
  v17 = (_BYTE *)sub_246F3F0(a1, v64);
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v64 + 8) + 8LL) - 17 <= 1 )
  {
    v70 = 257;
    v18 = sub_BCB2D0(v75);
    v19 = (_BYTE *)sub_ACD640(v18, 0, 0);
    v65 = (_BYTE *)sub_A837F0(&v71, v17, v19, (__int64)v69);
    if ( a3 != 1 )
    {
      v68 = 257;
      v20 = sub_BCB2D0(v75);
      v21 = (unsigned __int8 *)sub_ACD640(v20, 1, 0);
      v22 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v76 + 96LL);
      if ( v22 == sub_948070 )
      {
        if ( *v17 > 0x15u || *v21 > 0x15u )
          goto LABEL_40;
        v23 = (_BYTE *)sub_AD5840((__int64)v17, v21, 0);
      }
      else
      {
        v23 = (_BYTE *)v22(v76, v17, v21);
      }
      if ( v23 )
      {
LABEL_20:
        v70 = 257;
        v17 = (_BYTE *)sub_A82480(&v71, v65, v23, (__int64)v69);
        goto LABEL_21;
      }
LABEL_40:
      v70 = 257;
      v44 = sub_BD2C40(72, 2u);
      v23 = v44;
      if ( v44 )
        sub_B4DE80((__int64)v44, (__int64)v17, (__int64)v21, (__int64)v69, 0, 0);
      (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v77 + 16LL))(
        v77,
        v23,
        v67,
        v73,
        v74);
      v45 = v71;
      v46 = &v71[4 * v72];
      if ( v71 != v46 )
      {
        do
        {
          v47 = *((_QWORD *)v45 + 1);
          v48 = *v45;
          v45 += 4;
          sub_B99FD0((__int64)v23, v48, v47);
        }
        while ( v46 != v45 );
      }
      goto LABEL_20;
    }
    v17 = v65;
  }
LABEL_21:
  v24 = sub_246EE10(a1, v64);
  if ( *(_BYTE *)(a1 + 632) )
  {
    v69[1] = v24;
    v49 = *(unsigned int *)(a1 + 648);
    v50 = *(unsigned int *)(a1 + 652);
    v51 = *(_QWORD *)(a1 + 640);
    v69[0] = v17;
    v52 = (const __m128i *)v69;
    v53 = v49 + 1;
    v69[2] = a2;
    if ( v49 + 1 > v50 )
    {
      v57 = a1 + 640;
      v58 = (const void *)(a1 + 656);
      if ( v51 > (unsigned __int64)v69 || (unsigned __int64)v69 >= v51 + 24 * v49 )
      {
        sub_C8D5F0(v57, v58, v53, 0x18u, v53, v25);
        v51 = *(_QWORD *)(a1 + 640);
        v49 = *(unsigned int *)(a1 + 648);
      }
      else
      {
        v59 = (char *)v69 - v51;
        sub_C8D5F0(v57, v58, v53, 0x18u, v53, v25);
        v51 = *(_QWORD *)(a1 + 640);
        v49 = *(unsigned int *)(a1 + 648);
        v52 = (const __m128i *)&v59[v51];
      }
    }
    v26 = v61;
    v54 = (__m128i *)(v51 + 24 * v49);
    *v54 = _mm_loadu_si128(v52);
    v54[1].m128i_i64[0] = v52[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 648);
    if ( v61 )
      goto LABEL_23;
LABEL_53:
    v55 = sub_24637B0((__int64 *)a1, *((_QWORD *)a2 + 1));
    sub_246EF60(a1, (__int64)a2, v55);
    v56 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), (__int64)a2);
    v43 = (__int64)a2;
    sub_246F1C0(a1, (__int64)a2, v56);
    goto LABEL_39;
  }
  v26 = v61;
  if ( !v61 )
    goto LABEL_53;
LABEL_23:
  v27 = 0;
  v60 = (__int64)a2;
  v28 = sub_246F3F0(a1, v26);
  v66 = *(_QWORD *)(*(_QWORD *)(v28 + 8) + 24LL);
  do
  {
    while ( 1 )
    {
      v68 = 257;
      v30 = sub_BCB2D0(v75);
      v31 = (unsigned __int8 *)sub_ACD640(v30, v27, 0);
      v32 = (_BYTE *)sub_AD6530(v66, v27);
      v33 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v76 + 104LL);
      if ( v33 == sub_948040 )
        break;
      v29 = v33(v76, (_BYTE *)v28, v32, v31);
LABEL_27:
      if ( !v29 )
        goto LABEL_33;
      v28 = v29;
      if ( a3 == ++v27 )
        goto LABEL_38;
    }
    v34 = 0;
    if ( *(_BYTE *)v28 <= 0x15u )
      v34 = v28;
    if ( *v32 <= 0x15u && *v31 <= 0x15u && v34 )
    {
      v29 = sub_AD5A90(v34, v32, v31, 0);
      goto LABEL_27;
    }
LABEL_33:
    v70 = 257;
    v35 = sub_BD2C40(72, 3u);
    v37 = (__int64)v35;
    if ( v35 )
      sub_B4DFA0((__int64)v35, v28, (__int64)v32, (__int64)v31, (__int64)v69, v36, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v77 + 16LL))(
      v77,
      v37,
      v67,
      v73,
      v74);
    v38 = v71;
    v39 = &v71[4 * v72];
    if ( v71 != v39 )
    {
      do
      {
        v40 = *((_QWORD *)v38 + 1);
        v41 = *v38;
        v38 += 4;
        sub_B99FD0(v37, v41, v40);
      }
      while ( v39 != v38 );
    }
    v28 = v37;
    ++v27;
  }
  while ( a3 != v27 );
LABEL_38:
  sub_246EF60(a1, v60, v28);
  v42 = sub_246EE10(a1, v61);
  v43 = v60;
  sub_246F1C0(a1, v60, v42);
LABEL_39:
  sub_F94A20(&v71, v43);
}
