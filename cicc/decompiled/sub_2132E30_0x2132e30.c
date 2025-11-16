// Function: sub_2132E30
// Address: 0x2132e30
//
unsigned __int64 __fastcall sub_2132E30(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  char v14; // r10
  const void **v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  const void **v18; // r8
  unsigned __int8 v19; // r15
  int v20; // edx
  __int64 *v21; // r13
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int8 v24; // al
  unsigned int v25; // r15d
  __int64 v26; // rax
  char v27; // di
  const void **v28; // rax
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdi
  const void ***v33; // rdx
  unsigned int v34; // edx
  unsigned __int64 result; // rax
  unsigned int v36; // eax
  const void **v37; // r8
  char v38; // r10
  __int64 v39; // rdx
  unsigned int v40; // eax
  char v41; // r10
  const void **v42; // r8
  int v43; // r12d
  int v44; // eax
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned int v47; // r12d
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdi
  const void ***v55; // rdx
  unsigned int v56; // edx
  unsigned int v57; // eax
  int v58; // eax
  __int128 v59; // [rsp-20h] [rbp-E0h]
  __int128 v60; // [rsp-10h] [rbp-D0h]
  char v61; // [rsp+Fh] [rbp-B1h]
  char v62; // [rsp+Fh] [rbp-B1h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  __int64 v65; // [rsp+18h] [rbp-A8h]
  __int64 v66; // [rsp+18h] [rbp-A8h]
  const void **v67; // [rsp+18h] [rbp-A8h]
  const void **v68; // [rsp+18h] [rbp-A8h]
  __int64 v69; // [rsp+20h] [rbp-A0h]
  unsigned int v70; // [rsp+28h] [rbp-98h]
  const void **v71; // [rsp+28h] [rbp-98h]
  __int64 v72; // [rsp+60h] [rbp-60h] BYREF
  int v73; // [rsp+68h] [rbp-58h]
  char v74[8]; // [rsp+70h] [rbp-50h] BYREF
  const void **v75; // [rsp+78h] [rbp-48h]
  char v76[8]; // [rsp+80h] [rbp-40h] BYREF
  const void **v77; // [rsp+88h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 72);
  v72 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v72, v10, 2);
  v73 = *(_DWORD *)(a2 + 64);
  sub_20174B0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, (_DWORD *)a4);
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *a3;
  v13 = *(_QWORD *)(v11 + 40);
  v14 = *(_BYTE *)(v13 + 88);
  v15 = *(const void ***)(v13 + 96);
  v16 = *((unsigned int *)a3 + 2);
  v75 = v15;
  v17 = *(_QWORD *)(v12 + 40) + 16 * v16;
  v74[0] = v14;
  v18 = *(const void ***)(v17 + 8);
  v19 = *(_BYTE *)v17;
  v77 = v18;
  v76[0] = v19;
  if ( v19 == v14 )
  {
    if ( v19 || v18 == v15 )
      goto LABEL_5;
LABEL_35:
    v61 = v14;
    v63 = v11;
    v67 = v18;
    v57 = sub_1F58D40((__int64)v74);
    v38 = v61;
    v39 = v63;
    v70 = v57;
    v37 = v67;
    if ( !v19 )
      goto LABEL_36;
LABEL_18:
    v66 = v39;
    v40 = sub_2127930(v19);
    v11 = v66;
    goto LABEL_19;
  }
  if ( !v14 )
    goto LABEL_35;
  v65 = v11;
  v36 = sub_2127930(v14);
  v39 = v65;
  v70 = v36;
  if ( v19 )
    goto LABEL_18;
LABEL_36:
  v62 = v38;
  v64 = v39;
  v68 = v37;
  v40 = sub_1F58D40((__int64)v76);
  v41 = v62;
  v11 = v64;
  v18 = v68;
LABEL_19:
  if ( v40 < v70 )
  {
    if ( v41 )
    {
      v43 = sub_2127930(v41);
    }
    else
    {
      v71 = v18;
      v58 = sub_1F58D40((__int64)v74);
      v42 = v71;
      v43 = v58;
    }
    v76[0] = v19;
    v77 = v42;
    if ( v19 )
      v44 = sub_2127930(v19);
    else
      v44 = sub_1F58D40((__int64)v76);
    v47 = v43 - v44;
    v48 = *(_QWORD *)(a1 + 8);
    if ( v47 == 32 )
    {
      LOBYTE(v49) = 5;
    }
    else if ( v47 > 0x20 )
    {
      if ( v47 == 64 )
      {
        LOBYTE(v49) = 6;
      }
      else
      {
        if ( v47 != 128 )
        {
LABEL_38:
          v49 = sub_1F58CC0(*(_QWORD **)(v48 + 48), v47);
          v69 = v49;
          goto LABEL_29;
        }
        LOBYTE(v49) = 7;
      }
    }
    else if ( v47 == 8 )
    {
      LOBYTE(v49) = 3;
    }
    else
    {
      LOBYTE(v49) = 4;
      if ( v47 != 16 )
      {
        LOBYTE(v49) = 2;
        if ( v47 != 1 )
          goto LABEL_38;
      }
    }
    v50 = 0;
LABEL_29:
    v51 = v69;
    LOBYTE(v51) = v49;
    v52 = sub_1D2EF30((_QWORD *)v48, (unsigned int)v51, v50, v51, v45, v46);
    v54 = v53;
    v55 = (const void ***)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8));
    *((_QWORD *)&v60 + 1) = v54;
    *(_QWORD *)&v60 = v52;
    *(_QWORD *)a4 = sub_1D332F0(
                      (__int64 *)v48,
                      148,
                      (__int64)&v72,
                      *(unsigned __int8 *)v55,
                      v55[1],
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v60);
    result = v56;
    *(_DWORD *)(a4 + 8) = v56;
    goto LABEL_30;
  }
LABEL_5:
  *a3 = (__int64)sub_1D332F0(
                   *(__int64 **)(a1 + 8),
                   148,
                   (__int64)&v72,
                   v19,
                   v18,
                   0,
                   *(double *)a5.m128i_i64,
                   a6,
                   a7,
                   *a3,
                   a3[1],
                   *(_OWORD *)(v11 + 40));
  *((_DWORD *)a3 + 2) = v20;
  v21 = *(__int64 **)(a1 + 8);
  v22 = sub_1E0A0C0(v21[4]);
  v23 = 8 * sub_15A9520(v22, 0);
  if ( v23 == 32 )
  {
    v24 = 5;
  }
  else if ( v23 > 0x20 )
  {
    v24 = 6;
    if ( v23 != 64 )
    {
      v24 = 0;
      if ( v23 == 128 )
        v24 = 7;
    }
  }
  else
  {
    v24 = 3;
    if ( v23 != 8 )
      v24 = 4 * (v23 == 16);
  }
  v25 = v24;
  v26 = *(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8);
  v27 = *(_BYTE *)v26;
  v28 = *(const void ***)(v26 + 8);
  v76[0] = v27;
  v77 = v28;
  if ( v27 )
    v29 = sub_2127930(v27);
  else
    v29 = sub_1F58D40((__int64)v76);
  v30 = sub_1D38BB0((__int64)v21, (unsigned int)(v29 - 1), (__int64)&v72, v25, 0, 0, a5, a6, a7, 0);
  v32 = v31;
  v33 = (const void ***)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8));
  *((_QWORD *)&v59 + 1) = v32;
  *(_QWORD *)&v59 = v30;
  *(_QWORD *)a4 = sub_1D332F0(
                    v21,
                    123,
                    (__int64)&v72,
                    *(unsigned __int8 *)v33,
                    v33[1],
                    0,
                    *(double *)a5.m128i_i64,
                    a6,
                    a7,
                    *a3,
                    a3[1],
                    v59);
  result = v34;
  *(_DWORD *)(a4 + 8) = v34;
LABEL_30:
  if ( v72 )
    return sub_161E7C0((__int64)&v72, v72);
  return result;
}
