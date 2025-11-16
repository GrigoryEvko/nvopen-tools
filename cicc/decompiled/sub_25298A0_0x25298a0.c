// Function: sub_25298A0
// Address: 0x25298a0
//
__int64 __fastcall sub_25298A0(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v4; // r12
  __int128 v5; // kr00_16
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdi
  __m128i v10; // rax
  unsigned __int8 *v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rbx
  __m128i v19; // xmm2
  _QWORD *v20; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r13
  unsigned __int8 *v31; // rax
  __int64 v32; // r12
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rdi
  _BYTE *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r13
  int v48; // eax
  void (*v49)(); // rdx
  int v50; // r13d
  __int64 (__fastcall *v51)(__int64); // rax
  _BYTE *v52; // rdi
  void (*v53)(void); // rax
  __int64 v54; // [rsp+20h] [rbp-A0h]
  __int64 v55; // [rsp+28h] [rbp-98h]
  __int64 v56; // [rsp+28h] [rbp-98h]
  char v57; // [rsp+37h] [rbp-89h] BYREF
  __int64 v58; // [rsp+38h] [rbp-88h] BYREF
  __m128i v59; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v60[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v61; // [rsp+60h] [rbp-60h] BYREF
  _BYTE v62[72]; // [rsp+70h] [rbp-50h] BYREF

  sub_250D230((unsigned __int64 *)v62, (unsigned __int64)a2, 1, 0);
  v4 = *(_QWORD *)&v62[8];
  v5 = *(_OWORD *)v62;
  sub_250D230((unsigned __int64 *)v62, (unsigned __int64)a2, 5, 0);
  v6 = *a1;
  v59 = _mm_loadu_si128((const __m128i *)v62);
  sub_251BBC0(v6, v5, v4, 0, 2, 0, 1);
  v7 = *((_QWORD *)a2 - 4);
  if ( v7 && !*(_BYTE *)v7 )
  {
    v8 = v59.m128i_i64[0];
    sub_2522250(*a1, v59.m128i_i64[0], v59.m128i_i64[1], 0, 2, 0, 1);
    if ( !byte_4FEEF28 && sub_B2FC80(v7) )
    {
      if ( (*(_BYTE *)(v7 + 7) & 0x20) == 0 )
        return 1;
      v8 = 26;
      if ( !sub_B91C10(v7, 26) )
        return 1;
    }
    if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v7 + 24) + 16LL) + 8LL) != 7 )
    {
      if ( *((_QWORD *)a2 + 2) )
      {
        sub_250D230((unsigned __int64 *)v62, (unsigned __int64)a2, 3, 0);
        v9 = *a1;
        v8 = (__int64)v62;
        LOBYTE(v60[0]) = 0;
        v10.m128i_i64[0] = sub_2527850(v9, (__m128i *)v62, 0, v60, 1u);
        v61 = v10;
        if ( (unsigned __int8)sub_A750C0(**(_QWORD **)(*(_QWORD *)(v7 + 24) + 16LL)) )
        {
          v8 = v5;
          sub_2521600(*a1, v5, *((__int64 *)&v5 + 1), 0, 2, 0, 1);
        }
      }
    }
    v11 = (unsigned __int8 *)(v59.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL);
    if ( (v59.m128i_i8[0] & 3) == 3 )
    {
      v11 = (unsigned __int8 *)*((_QWORD *)v11 + 3);
      v12 = *v11;
      if ( (unsigned __int8)v12 <= 0x1Cu )
        goto LABEL_10;
    }
    else
    {
      v12 = *v11;
      if ( (unsigned __int8)v12 <= 0x1Cu )
        goto LABEL_10;
    }
    v16 = (unsigned int)(v12 - 34);
    if ( (unsigned __int8)v16 <= 0x33u )
    {
      v17 = 0x8000000000041LL;
      if ( _bittest64(&v17, v16) )
      {
        v13 = *((_QWORD *)v11 + 9);
LABEL_11:
        v14 = *a2;
        v58 = v13;
        if ( v14 == 40 )
        {
          v15 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
        }
        else
        {
          v15 = 0;
          if ( v14 != 85 )
          {
            v15 = 64;
            if ( v14 != 34 )
              BUG();
          }
        }
        if ( (a2[7] & 0x80u) != 0 )
        {
          v22 = sub_BD2BC0((__int64)a2);
          v24 = v22 + v23;
          if ( (a2[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v24 >> 4) )
              goto LABEL_78;
          }
          else if ( (unsigned int)((v24 - sub_BD2BC0((__int64)a2)) >> 4) )
          {
            if ( (a2[7] & 0x80u) != 0 )
            {
              v25 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
              if ( (a2[7] & 0x80u) == 0 )
                BUG();
              v26 = sub_BD2BC0((__int64)a2);
              v28 = 32LL * (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v25);
              goto LABEL_36;
            }
LABEL_78:
            BUG();
          }
        }
        v28 = 0;
LABEL_36:
        v29 = (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v15 - v28) >> 5;
        if ( (int)v29 > 0 )
        {
          v30 = 0;
          v54 = (unsigned int)(v29 - 1);
          while ( 1 )
          {
            v31 = (a2[7] & 0x40) != 0
                ? (unsigned __int8 *)*((_QWORD *)a2 - 1)
                : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            *(_OWORD *)v62 = (unsigned __int64)&v31[32 * v30] | 3;
            nullsub_1518();
            v32 = sub_A744E0(&v58, v30);
            sub_251BBC0(*a1, *(__int64 *)v62, *(__int64 *)&v62[8], 0, 2, 0, 1);
            v33 = *a1;
            v57 = 0;
            v34 = sub_2527850(v33, (__m128i *)v62, 0, &v57, 1u);
            v35 = *a1;
            v60[1] = v36;
            v60[0] = v34;
            sub_251F380(v35, (const __m128i *)v62, v32, 0);
            v37 = *(_QWORD *)(*(_QWORD *)&a2[32 * (v30 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))] + 8LL);
            if ( *(_BYTE *)(v37 + 8) == 14 )
            {
              sub_251F740(*a1, (const __m128i *)v62, v32, 0);
              sub_25284B0(*a1, (__int64 *)v62, v32, 1);
              sub_251FA90(*a1, (const __m128i *)v62, v32, 0);
              sub_2521A40(*a1, *(__int64 *)v62, *(__int64 *)&v62[8], 0, 2, 0, 1);
              sub_2521E40(*a1, *(__int64 *)v62, *(__int64 *)&v62[8], 0, 2, 0, 1);
              if ( !(unsigned __int8)sub_A74710(&v58, (int)v30 + 1, 50) )
                sub_25294B0(*a1, *(__int64 *)v62, *(__int64 *)&v62[8], 0, 2, 0, 1);
              sub_2520670(*a1, (__m128i *)v62, v32, 0);
            }
            else if ( (unsigned __int8)sub_A750C0(v37) )
            {
              sub_2521600(*a1, *(__int64 *)v62, *(__int64 *)&v62[8], 0, 2, 0, 1);
            }
            if ( v30 == v54 )
              break;
            ++v30;
          }
        }
        return 1;
      }
    }
LABEL_10:
    v13 = *((_QWORD *)sub_250CBE0(v59.m128i_i64, v8) + 15);
    goto LABEL_11;
  }
  v18 = *a1;
  v61 = _mm_loadu_si128(&v59);
  if ( !(unsigned __int8)sub_250E300(v18, &v61) )
    v61.m128i_i64[1] = 0;
  v19 = _mm_loadu_si128(&v61);
  *(_QWORD *)v62 = &unk_438A65A;
  *(__m128i *)&v62[8] = v19;
  v20 = sub_25134D0(v18 + 136, (__int64 *)v62);
  if ( (!v20 || !v20[3]) && (unsigned __int8)sub_2509800(&v61) == 5 )
  {
    v38 = (_BYTE *)sub_2509740(&v61);
    if ( *v38 == 85 )
    {
      v55 = (__int64)v38;
      if ( sub_B491E0((__int64)v38) && !sub_B49200(v55) )
      {
        v39 = *(_QWORD *)(v18 + 4376);
        if ( !v39 || (*(_QWORD *)v62 = &unk_438A65A, sub_2517B80(v39, (__int64 *)v62)) )
        {
          v40 = sub_25096F0(&v61);
          if ( !v40 || (v56 = v40, !(unsigned __int8)sub_B2D610(v40, 20)) && !(unsigned __int8)sub_B2D610(v56, 48) )
          {
            if ( (unsigned __int8)sub_250CDD0(v18, v61.m128i_i64, (char *)v60) )
            {
              v41 = sub_2565F60(&v61, v18);
              *(_QWORD *)v62 = &unk_438A65A;
              v42 = v41;
              *(__m128i *)&v62[8] = _mm_loadu_si128((const __m128i *)(v41 + 72));
              *sub_2519B70(v18 + 136, (__int64)v62) = v41;
              if ( *(_DWORD *)(v18 + 3552) <= 1u )
              {
                *(_QWORD *)v62 = v42 & 0xFFFFFFFFFFFFFFFBLL;
                sub_251B630(v18 + 224, (unsigned __int64 *)v62, v43, v44, v45, v46);
                if ( !*(_DWORD *)(v18 + 3552) && !(unsigned __int8)sub_250E880(v18, v42) )
                  goto LABEL_70;
              }
              *(_QWORD *)v62 = v42;
              v47 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BD00, (__int64)v62);
              v48 = *(_DWORD *)(v18 + 3556);
              *(_DWORD *)(v18 + 3556) = v48 + 1;
              v49 = *(void (**)())(*(_QWORD *)v42 + 24LL);
              if ( v49 != nullsub_1516 )
              {
                ((void (__fastcall *)(__int64, __int64))v49)(v42, v18);
                v48 = *(_DWORD *)(v18 + 3556) - 1;
              }
              *(_DWORD *)(v18 + 3556) = v48;
              if ( v47 )
                sub_C9AF60(v47);
              if ( LOBYTE(v60[0]) )
              {
                v50 = *(_DWORD *)(v18 + 3552);
                *(_DWORD *)(v18 + 3552) = 1;
                sub_251C580(v18, v42);
                *(_DWORD *)(v18 + 3552) = v50;
              }
              else
              {
LABEL_70:
                v51 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v42 + 40LL);
                if ( v51 == sub_2505F20 )
                  v52 = (_BYTE *)(v42 + 88);
                else
                  v52 = (_BYTE *)v51(v42);
                v53 = *(void (**)(void))(*(_QWORD *)v52 + 40LL);
                if ( (char *)v53 == (char *)sub_2505E20 )
                  v52[9] = v52[8];
                else
                  v53();
              }
            }
          }
        }
      }
    }
  }
  return 1;
}
