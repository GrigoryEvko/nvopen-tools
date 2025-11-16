// Function: sub_20A2630
// Address: 0x20a2630
//
__int64 __fastcall sub_20A2630(
        double a1,
        double a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned int a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v9; // r13
  unsigned int v10; // r15d
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rdx
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // r14d
  __int64 v22; // r9
  int v23; // eax
  unsigned int v24; // r11d
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // r11
  __int64 v29; // r10
  __int64 v30; // r9
  unsigned int v31; // eax
  const void **v32; // r13
  __int64 v33; // rdx
  __int64 (*v34)(); // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  const void **v37; // rdx
  char v38; // al
  __int64 (*v39)(); // rax
  __int64 v40; // rdx
  unsigned int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int128 v45; // rax
  __int64 v46; // rax
  int v47; // edx
  __int64 v48; // [rsp+0h] [rbp-B0h]
  __int64 v49; // [rsp+0h] [rbp-B0h]
  __int64 v50; // [rsp+0h] [rbp-B0h]
  __int64 v51; // [rsp+0h] [rbp-B0h]
  __int64 v52; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+8h] [rbp-A8h]
  __int64 v54; // [rsp+8h] [rbp-A8h]
  int v55; // [rsp+10h] [rbp-A0h]
  __int64 v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+20h] [rbp-90h]
  __int128 v63; // [rsp+30h] [rbp-80h]
  __int64 v64; // [rsp+60h] [rbp-50h] BYREF
  int v65; // [rsp+68h] [rbp-48h]
  _BYTE v66[8]; // [rsp+70h] [rbp-40h] BYREF
  __int64 v67; // [rsp+78h] [rbp-38h]

  v13 = *(_QWORD *)(a5 + 72);
  v60 = *a9;
  v64 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v64, v13, 2);
  v14 = 16LL * a6;
  v15 = v14 + *(_QWORD *)(a5 + 40);
  v65 = *(_DWORD *)(a5 + 64);
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v66[0] = v16;
  v67 = v17;
  if ( v16 )
    v18 = (unsigned __int8)(v16 - 14) <= 0x5Fu;
  else
    v18 = sub_1F58D20((__int64)v66);
  if ( v18 )
    goto LABEL_7;
  v19 = *(_QWORD *)(a5 + 48);
  if ( !v19 || *(_QWORD *)(v19 + 32) )
    goto LABEL_7;
  v22 = *(_QWORD *)(v60 + 16);
  if ( *(_DWORD *)(a8 + 8) <= 0x40u )
  {
    if ( !*(_QWORD *)a8 )
    {
      v24 = 0;
      goto LABEL_15;
    }
    _BitScanReverse64(&v36, *(_QWORD *)a8);
    v24 = 64 - (v36 ^ 0x3F);
  }
  else
  {
    v52 = *(_QWORD *)(v60 + 16);
    v55 = *(_DWORD *)(a8 + 8);
    v23 = sub_16A57B0(a8);
    v22 = v52;
    v24 = v55 - v23;
    if ( v55 == v23 )
    {
LABEL_15:
      v24 = (((((((((v24 | ((unsigned __int64)v24 >> 1)) >> 2) | v24 | ((unsigned __int64)v24 >> 1)) >> 4)
               | ((v24 | ((unsigned __int64)v24 >> 1)) >> 2)
               | v24
               | ((unsigned __int64)v24 >> 1)) >> 8)
             | ((((v24 | ((unsigned __int64)v24 >> 1)) >> 2) | v24 | ((unsigned __int64)v24 >> 1)) >> 4)
             | ((v24 | ((unsigned __int64)v24 >> 1)) >> 2)
             | v24
             | ((unsigned __int64)v24 >> 1)) >> 16)
           | ((((((v24 | ((unsigned __int64)v24 >> 1)) >> 2) | v24 | ((unsigned __int64)v24 >> 1)) >> 4)
             | ((v24 | ((unsigned __int64)v24 >> 1)) >> 2)
             | v24
             | ((unsigned __int64)v24 >> 1)) >> 8)
           | ((((v24 | ((unsigned __int64)v24 >> 1)) >> 2) | v24 | ((unsigned __int64)v24 >> 1)) >> 4)
           | ((v24 | ((unsigned __int64)v24 >> 1)) >> 2)
           | v24
           | (v24 >> 1))
          + 1;
      goto LABEL_16;
    }
  }
  if ( (v24 & (v24 - 1)) != 0 )
    goto LABEL_15;
LABEL_16:
  if ( a7 <= v24 )
  {
LABEL_7:
    v20 = 0;
    goto LABEL_8;
  }
  v25 = a5;
  v26 = v24;
  v27 = v22;
  v28 = v48;
  v29 = v9;
  v30 = v25;
  while ( 1 )
  {
    if ( (_DWORD)v26 == 32 )
    {
      LOBYTE(v31) = 5;
    }
    else if ( (unsigned int)v26 <= 0x20 )
    {
      if ( (_DWORD)v26 == 8 )
      {
        LOBYTE(v31) = 3;
      }
      else
      {
        LOBYTE(v31) = 4;
        if ( (_DWORD)v26 != 16 )
        {
          LOBYTE(v31) = 2;
          if ( (_DWORD)v26 != 1 )
          {
LABEL_33:
            v49 = v29;
            v53 = v28;
            v56 = v30;
            v31 = sub_1F58CC0(*(_QWORD **)(v60 + 48), v26);
            v29 = v49;
            v28 = v53;
            v30 = v56;
            v10 = v31;
            v32 = v37;
            goto LABEL_21;
          }
        }
      }
    }
    else if ( (_DWORD)v26 == 64 )
    {
      LOBYTE(v31) = 6;
    }
    else
    {
      if ( (_DWORD)v26 != 128 )
        goto LABEL_33;
      LOBYTE(v31) = 7;
    }
    v32 = 0;
LABEL_21:
    LOBYTE(v10) = v31;
    v33 = v14 + *(_QWORD *)(v30 + 40);
    v34 = *(__int64 (**)())(*(_QWORD *)v27 + 800LL);
    LOBYTE(v29) = *(_BYTE *)v33;
    if ( v34 != sub_1D12DF0 )
    {
      v50 = v28;
      v54 = v30;
      v57 = v29;
      v38 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, const void **))v34)(
              v27,
              (unsigned int)v29,
              *(_QWORD *)(v33 + 8),
              v10,
              v32);
      v29 = v57;
      v30 = v54;
      v28 = v50;
      if ( v38 )
      {
        v39 = *(__int64 (**)())(*(_QWORD *)v27 + 824LL);
        v40 = v14 + *(_QWORD *)(v54 + 40);
        LOBYTE(v28) = *(_BYTE *)v40;
        if ( v39 != sub_1D12E00 )
        {
          v51 = v57;
          v58 = v28;
          v41 = ((__int64 (__fastcall *)(__int64, _QWORD, const void **, _QWORD, _QWORD, __int64))v39)(
                  v27,
                  v10,
                  v32,
                  (unsigned int)v28,
                  *(_QWORD *)(v40 + 8),
                  v54);
          v28 = v58;
          v30 = v54;
          v29 = v51;
          if ( (_BYTE)v41 )
            break;
        }
      }
    }
    v35 = (((((((((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 4) | ((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 8)
           | ((((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 4)
           | ((v26 | (v26 >> 1)) >> 2)
           | v26
           | (v26 >> 1)) >> 16)
         | ((((((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 4) | ((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 8)
         | ((((v26 | (v26 >> 1)) >> 2) | v26 | (v26 >> 1)) >> 4)
         | ((v26 | (v26 >> 1)) >> 2)
         | v26
         | (v26 >> 1))
        + 1;
    v26 = (unsigned int)v35;
    if ( a7 <= (unsigned int)v35 )
      goto LABEL_7;
  }
  v20 = v41;
  *(_QWORD *)&v63 = sub_1D309E0(
                      (__int64 *)v60,
                      145,
                      (__int64)&v64,
                      v10,
                      v32,
                      0,
                      a1,
                      a2,
                      *(double *)a3.m128i_i64,
                      *(_OWORD *)(*(_QWORD *)(v54 + 32) + 40LL));
  *((_QWORD *)&v63 + 1) = v42;
  v43 = sub_1D309E0(
          (__int64 *)v60,
          145,
          (__int64)&v64,
          v10,
          v32,
          0,
          a1,
          a2,
          *(double *)a3.m128i_i64,
          *(_OWORD *)*(_QWORD *)(v54 + 32));
  *(_QWORD *)&v45 = sub_1D332F0(
                      (__int64 *)v60,
                      *(unsigned __int16 *)(v54 + 24),
                      (__int64)&v64,
                      v10,
                      v32,
                      0,
                      a1,
                      a2,
                      a3,
                      v43,
                      v44,
                      v63);
  v46 = sub_1D309E0(
          (__int64 *)v60,
          144,
          (__int64)&v64,
          *(unsigned __int8 *)(*(_QWORD *)(v54 + 40) + 16LL * a6),
          *(const void ***)(*(_QWORD *)(v54 + 40) + 16LL * a6 + 8),
          0,
          a1,
          a2,
          *(double *)a3.m128i_i64,
          v45);
  a9[2] = v54;
  *((_DWORD *)a9 + 6) = a6;
  a9[4] = v46;
  *((_DWORD *)a9 + 10) = v47;
LABEL_8:
  if ( v64 )
    sub_161E7C0((__int64)&v64, v64);
  return v20;
}
