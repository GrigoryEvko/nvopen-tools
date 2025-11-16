// Function: sub_28F9120
// Address: 0x28f9120
//
void __fastcall sub_28F9120(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  __int64 v9; // r14
  __int32 v10; // eax
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // rdx
  __int32 v14; // r13d
  unsigned __int64 v15; // r9
  __m128i *v16; // rax
  __int64 v17; // rdx
  unsigned __int8 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 *v22; // r14
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __m128i *v25; // rdi
  int v26; // r13d
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  char v34; // r13
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  int v39; // r15d
  unsigned int v40; // r13d
  __int64 v41; // r12
  __m128i *v42; // r11
  __m128i *v43; // rdi
  __int64 v44; // rbx
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  int v47; // r14d
  unsigned int i; // eax
  __int64 v49; // rcx
  unsigned int v50; // eax
  unsigned int v51; // edx
  unsigned __int32 v52; // eax
  __int64 v53; // r15
  __int64 v54; // r14
  __int64 v55; // r13
  __m128i *v56; // rsi
  __int64 v57; // rbx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  _BYTE *v64; // rax
  unsigned int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v70; // [rsp+8h] [rbp-1F8h]
  unsigned int v71; // [rsp+14h] [rbp-1ECh]
  unsigned int v72; // [rsp+20h] [rbp-1E0h]
  unsigned int v73; // [rsp+24h] [rbp-1DCh]
  unsigned int v74; // [rsp+28h] [rbp-1D8h]
  unsigned int v75; // [rsp+2Ch] [rbp-1D4h]
  __int64 v76; // [rsp+30h] [rbp-1D0h]
  int v77; // [rsp+30h] [rbp-1D0h]
  _BYTE *v79; // [rsp+58h] [rbp-1A8h]
  _DWORD *v80; // [rsp+58h] [rbp-1A8h]
  unsigned __int32 v81; // [rsp+58h] [rbp-1A8h]
  __int64 v82; // [rsp+58h] [rbp-1A8h]
  unsigned int v83; // [rsp+6Ch] [rbp-194h] BYREF
  __int64 v84[4]; // [rsp+70h] [rbp-190h] BYREF
  __int64 v85[4]; // [rsp+90h] [rbp-170h] BYREF
  _BYTE *v86; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-148h]
  _BYTE v88[128]; // [rsp+C0h] [rbp-140h] BYREF
  __m128i *v89; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+148h] [rbp-B8h]
  _BYTE v91[176]; // [rsp+150h] [rbp-B0h] BYREF

  v2 = a1;
  v86 = v88;
  v87 = 0x800000000LL;
  v83 = (unsigned int)&loc_1010101;
  v76 = a1 + 64;
  v3 = sub_28F1DF0(a2, (__int64)&v86, a1 + 64, &v83);
  v6 = (unsigned int)v87;
  *(_BYTE *)(a1 + 752) |= v3;
  v89 = (__m128i *)v91;
  v90 = 0x800000000LL;
  if ( v6 > 8 )
  {
    sub_C8D5F0((__int64)&v89, v91, v6, 0x10u, v4, v5);
    v6 = (unsigned int)v87;
  }
  v7 = 16 * v6;
  if ( v86 != &v86[v7] )
  {
    v79 = &v86[v7];
    v8 = (unsigned __int64)v86;
    do
    {
      v9 = *(_QWORD *)v8;
      v10 = sub_28EF780(a1, *(_BYTE **)v8);
      v12 = *(_QWORD *)(v8 + 8);
      v13 = (unsigned int)v90;
      v14 = v10;
      v15 = v12 + (unsigned int)v90;
      if ( v15 > HIDWORD(v90) )
      {
        sub_C8D5F0((__int64)&v89, v91, v12 + (unsigned int)v90, 0x10u, v11, v15);
        v13 = (unsigned int)v90;
      }
      v16 = &v89[v13];
      if ( v12 )
      {
        v17 = v12;
        do
        {
          if ( v16 )
          {
            v16->m128i_i32[0] = v14;
            v16->m128i_i64[1] = v9;
          }
          ++v16;
          --v17;
        }
        while ( v17 );
        LODWORD(v13) = v90;
      }
      v8 += 16LL;
      LODWORD(v90) = v13 + v12;
    }
    while ( v79 != (_BYTE *)v8 );
  }
  sub_28ED9A0((__int64)&v89);
  v18 = sub_28F8EB0(a1, a2, &v89);
  v20 = (__int64)v18;
  if ( v18 )
  {
    if ( a2 != v18 )
    {
      sub_BD84D0((__int64)a2, (__int64)v18);
      if ( *(_BYTE *)v20 > 0x1Cu )
      {
        v21 = *((_QWORD *)a2 + 6);
        if ( v21 )
        {
          v22 = (__int64 *)(v20 + 48);
          v85[0] = *((_QWORD *)a2 + 6);
          sub_B96E90((__int64)v85, v21, 1);
          if ( (__int64 *)(v20 + 48) != v85 )
            goto LABEL_19;
          goto LABEL_40;
        }
      }
      goto LABEL_23;
    }
    v25 = v89;
    goto LABEL_25;
  }
  v26 = v90;
  v27 = *((_QWORD *)a2 + 2);
  if ( v27 && !*(_QWORD *)(v27 + 8) )
  {
    if ( *a2 == 46 )
    {
      if ( **(_BYTE **)(v27 + 24) == 42 )
      {
        v64 = (_BYTE *)v89[(unsigned int)v90 - 1].m128i_i64[1];
        if ( *v64 == 17 && sub_986760((__int64)(v64 + 24)) )
          goto LABEL_96;
      }
    }
    else if ( *a2 == 47 && **(_BYTE **)(v27 + 24) == 43 )
    {
      v30 = v89[(unsigned int)v90 - 1].m128i_i64[1];
      if ( *(_BYTE *)v30 == 18 )
      {
        v80 = sub_C33320();
        sub_C3B1B0((__int64)v85, -1.0);
        sub_C407B0(v84, v85, v80);
        sub_C338F0((__int64)v85);
        sub_C41640(v84, *(_DWORD **)(v30 + 24), 1, (bool *)v85);
        v34 = sub_AC3090(v30, v84, v31, v32, v33);
        sub_91D830(v84);
        if ( !v34 )
        {
LABEL_48:
          v26 = v90;
          goto LABEL_31;
        }
LABEL_96:
        v65 = sub_28EDA80((__int64)&v89);
        sub_28EAC10((__int64)&v89, v89, v65, v66, v67, v68);
        goto LABEL_48;
      }
    }
  }
LABEL_31:
  if ( v26 == 1 )
  {
    v25 = v89;
    v28 = v89->m128i_i64[1];
    if ( (unsigned __int8 *)v28 != a2 )
    {
      sub_BD84D0((__int64)a2, v28);
      v20 = v89->m128i_i64[1];
      if ( *(_BYTE *)v20 > 0x1Cu )
      {
        v29 = *((_QWORD *)a2 + 6);
        v22 = (__int64 *)(v20 + 48);
        v85[0] = v29;
        if ( v29 )
        {
          sub_B96E90((__int64)v85, v29, 1);
          if ( v22 != v85 )
          {
LABEL_19:
            v23 = *(_QWORD *)(v20 + 48);
            if ( !v23 )
              goto LABEL_21;
            goto LABEL_20;
          }
LABEL_40:
          if ( v85[0] )
            sub_B91220((__int64)v85, v85[0]);
          goto LABEL_23;
        }
        if ( v22 != v85 )
        {
          v23 = *(_QWORD *)(v20 + 48);
          if ( v23 )
          {
LABEL_20:
            sub_B91220((__int64)v22, v23);
LABEL_21:
            v24 = (unsigned __int8 *)v85[0];
            *(_QWORD *)(v20 + 48) = v85[0];
            if ( v24 )
              sub_B976B0((__int64)v85, v24, (__int64)v22);
          }
        }
      }
LABEL_23:
      sub_D68D20((__int64)v85, 0, (__int64)a2);
      sub_28F19A0(v76, v85);
      sub_D68D70(v85);
      v25 = v89;
    }
LABEL_25:
    if ( v25 == (__m128i *)v91 )
      goto LABEL_27;
    goto LABEL_26;
  }
  if ( (unsigned int)(v26 - 3) > 7 || (_BYTE)qword_5004D28 == 1 )
    goto LABEL_34;
  v19 = (__int64)a2;
  if ( (_BYTE)qword_5004C48 )
  {
    v35 = (unsigned int)(v26 - 2);
    while ( 1 )
    {
      v37 = v89[v35].m128i_i64[1];
      if ( *(_BYTE *)v37 > 0x1Cu )
        break;
      v38 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)a2 + 5) + 72LL) + 80LL);
      if ( v38 )
      {
        v36 = v38 - 24;
LABEL_52:
        if ( v20 )
        {
          if ( v36 != v20 )
            goto LABEL_91;
        }
        else
        {
          v20 = v36;
        }
        goto LABEL_54;
      }
      if ( v20 )
      {
LABEL_91:
        v39 = v35 + 1;
        v71 = v35 + 1;
        v72 = v26 - 1;
        if ( v26 - 1 > (unsigned int)(v35 + 1) )
          goto LABEL_59;
        goto LABEL_34;
      }
LABEL_54:
      if ( (_DWORD)--v35 == -1 )
        goto LABEL_58;
    }
    v36 = *(_QWORD *)(v37 + 40);
    goto LABEL_52;
  }
LABEL_58:
  v39 = 0;
  v71 = 0;
  v72 = v26 - 1;
LABEL_59:
  v40 = 1;
  v73 = 0;
  v70 = a1 + 32LL * ((unsigned int)*a2 - 42) + 176;
  v81 = 0;
  v74 = 0;
  do
  {
    v75 = v72--;
    v19 = v72;
    if ( (int)v72 >= v39 )
    {
      v41 = *(_QWORD *)(v70 + 8);
      v42 = &v89[v75];
      v43 = &v89[v72];
      v44 = *(unsigned int *)(v70 + 24);
      v77 = v44 - 1;
      do
      {
        v45 = v43->m128i_u64[1];
        v46 = v42->m128i_u64[1];
        if ( v45 < v46 )
        {
          v46 = v43->m128i_u64[1];
          v45 = v42->m128i_u64[1];
        }
        if ( (_DWORD)v44 )
        {
          v47 = 1;
          for ( i = v77
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)
                      | ((unsigned __int64)(((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)))); ; i = v77 & v50 )
          {
            v49 = v41 + 72LL * i;
            if ( v46 == *(_QWORD *)v49 && v45 == *(_QWORD *)(v49 + 8) )
              break;
            if ( *(_QWORD *)v49 == -4096 && *(_QWORD *)(v49 + 8) == -4096 )
              goto LABEL_77;
            v50 = v47 + i;
            ++v47;
          }
          if ( v41 + 72 * v44 != v49 && *(_QWORD *)(v49 + 32) && *(_QWORD *)(v49 + 56) )
          {
            v51 = *(_DWORD *)(v49 + 64);
            v52 = v43->m128i_i32[0];
            if ( v42->m128i_i32[0] >= (unsigned __int32)v43->m128i_i32[0] )
              v52 = v42->m128i_i32[0];
            if ( v51 > v40 || v51 == v40 && v81 > v52 )
            {
              v74 = v19;
              v40 = *(_DWORD *)(v49 + 64);
              v81 = v52;
              v73 = v75;
            }
          }
        }
LABEL_77:
        v19 = (unsigned int)(v19 - 1);
        --v43;
      }
      while ( v39 <= (int)v19 );
    }
  }
  while ( v72 > v71 );
  v2 = a1;
  if ( v40 != 1 )
  {
    v53 = v74;
    v54 = v89[v53].m128i_i64[0];
    v55 = v89[v53].m128i_i64[1];
    v56 = &v89[v73];
    v57 = v56->m128i_i64[1];
    v82 = v56->m128i_i64[0];
    sub_28E9AE0((__int64)&v89, v56->m128i_i8);
    sub_28E9AE0((__int64)&v89, v89[v53].m128i_i8);
    sub_28ED700((__int64)&v89, v54, v55, v58, v59, v60);
    sub_28ED700((__int64)&v89, v82, v57, v61, v62, v63);
  }
LABEL_34:
  sub_28F62D0(v2, a2, (__int64)&v89, v83, v19);
  v25 = v89;
  if ( v89 != (__m128i *)v91 )
LABEL_26:
    _libc_free((unsigned __int64)v25);
LABEL_27:
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
}
