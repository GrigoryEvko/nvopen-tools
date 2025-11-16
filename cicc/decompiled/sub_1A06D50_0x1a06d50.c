// Function: sub_1A06D50
// Address: 0x1a06d50
//
__int64 __fastcall sub_1A06D50(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v11; // r13
  int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // r14
  void *v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // rax
  int v23; // r14d
  __int64 v24; // r12
  __int64 v25; // rdx
  char v26; // al
  int v27; // r14d
  __int64 v28; // rax
  __int64 *v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 *v33; // rdx
  __int64 v34; // rsi
  unsigned __int64 v35; // rcx
  __int64 v36; // rcx
  __int64 **v37; // rdx
  __int64 *v38; // rsi
  __int64 **v39; // rax
  __int64 v40; // rdx
  int v41; // edi
  int v42; // esi
  __int64 v43; // rdx
  __int64 v44; // rdi
  __int64 *v45; // r12
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rsi
  int v49; // edi
  unsigned int v50; // ecx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // rsi
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v59; // rsi
  __int64 v60; // r12
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 v63; // rdx
  unsigned int v64; // ecx
  void *v65; // [rsp+8h] [rbp-98h]
  __int64 v66; // [rsp+8h] [rbp-98h]
  void *v67; // [rsp+10h] [rbp-90h]
  __int64 v69; // [rsp+18h] [rbp-88h]
  _QWORD v70[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v71[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v72; // [rsp+40h] [rbp-60h]
  char v73[8]; // [rsp+50h] [rbp-50h] BYREF
  void *v74; // [rsp+58h] [rbp-48h] BYREF
  __int64 v75; // [rsp+60h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 8);
  if ( !v10 )
    return 0;
  v11 = *(_QWORD *)(v10 + 8);
  if ( v11 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    return 0;
  v13 = *(unsigned __int8 *)(a2 + 16);
  if ( v13 != 40 && v13 != 43 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(__int64 **)(a2 - 8);
  else
    v14 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v15 = *v14;
  v16 = v14[3];
  v17 = *(_BYTE *)(v16 + 16);
  if ( *(_BYTE *)(v15 + 16) == 14 )
  {
    if ( v17 == 14 )
      return 0;
    v11 = v15;
  }
  else
  {
    v15 = v16;
    if ( v17 != 14 )
      return 0;
  }
  v20 = *(void **)(v15 + 32);
  v65 = v20;
  v67 = sub_16982C0();
  if ( v20 == v67 )
    v21 = *(_QWORD *)(v15 + 40) + 8LL;
  else
    v21 = v15 + 32;
  if ( (*(_BYTE *)(v21 + 18) & 8) == 0 )
    return 0;
  v22 = sub_1648700(v10);
  v23 = *((unsigned __int8 *)v22 + 16);
  v24 = (__int64)v22;
  v25 = (unsigned int)(v23 - 35);
  v26 = *((_BYTE *)v22 + 16);
  if ( (unsigned int)v25 > 0x11 )
    return 0;
  if ( !*(_QWORD *)(v24 + 8) )
    return 0;
  v27 = v23 - 24;
  if ( (v26 & 0xFD) != 0x24 )
    return 0;
  if ( ((1LL << v27) & 0x1C019800) == 0 )
  {
    if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
    {
      v28 = *(_QWORD *)(v24 - 8);
    }
    else
    {
      v25 = 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF);
      v28 = v24 - v25;
    }
    if ( a2 != *(_QWORD *)(v28 + 24) )
      return 0;
  }
  if ( v27 == 12 )
  {
    if ( !sub_15FB6B0(v24, a2, v25, 12) && !sub_15FB6D0(v24, 0, v53, v54) && sub_1A014D0(v24) )
      return 0;
    v65 = *(void **)(v15 + 32);
  }
  v29 = (__int64 *)(v15 + 32);
  if ( v67 == v65 )
    sub_169C6E0(&v74, (__int64)v29);
  else
    sub_16986C0(&v74, v29);
  if ( v67 == v74 )
    sub_169C8D0((__int64)&v74, *(double *)a3.m128_u64, a4, a5);
  else
    sub_1699490((__int64)&v74);
  v30 = (_QWORD *)sub_16498A0(v15);
  v31 = sub_159CCF0(v30, (__int64)v73);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v32 = *(_QWORD *)(a2 - 8);
  else
    v32 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v33 = (__int64 *)(v32 + 24LL * (v11 == 0));
  if ( *v33 )
  {
    v34 = v33[1];
    v35 = v33[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v35 = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 16) = *(_QWORD *)(v34 + 16) & 3LL | v35;
  }
  *v33 = v31;
  if ( v31 )
  {
    v36 = *(_QWORD *)(v31 + 8);
    v33[1] = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 16) = (unsigned __int64)(v33 + 1) | *(_QWORD *)(v36 + 16) & 3LL;
    v33[2] = (v31 + 8) | v33[2] & 3;
    *(_QWORD *)(v31 + 8) = v33;
  }
  if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
  {
    v37 = *(__int64 ***)(v24 - 8);
    v38 = *v37;
    v39 = v37;
    if ( (__int64 *)a2 != *v37 )
      goto LABEL_40;
    v64 = *(unsigned __int8 *)(v24 + 16) - 24;
    if ( v64 > 0x1C || ((1LL << v64) & 0x1C019800) == 0 )
      goto LABEL_93;
  }
  else
  {
    v49 = *(_DWORD *)(v24 + 20);
    v39 = (__int64 **)(v24 - 24LL * (v49 & 0xFFFFFFF));
    v38 = *v39;
    if ( (__int64 *)a2 != *v39 )
      goto LABEL_40;
    v50 = *(unsigned __int8 *)(v24 + 16) - 24;
    if ( v50 > 0x1C || ((1LL << v50) & 0x1C019800) == 0 )
      goto LABEL_55;
  }
  sub_15FB800(v24);
  if ( (*(_BYTE *)(v24 + 23) & 0x40) == 0 )
  {
    v49 = *(_DWORD *)(v24 + 20);
LABEL_55:
    v39 = (__int64 **)(v24 - 24LL * (v49 & 0xFFFFFFF));
    v38 = *v39;
    goto LABEL_40;
  }
  v37 = *(__int64 ***)(v24 - 8);
LABEL_93:
  v38 = *v37;
  v39 = v37;
LABEL_40:
  v40 = (__int64)v39[3];
  v72 = 257;
  v41 = 14;
  if ( v27 != 12 )
    v41 = 12;
  v18 = sub_15FB440(v41, v38, v40, (__int64)v71, 0);
  v42 = *(_BYTE *)(v24 + 17) >> 1;
  if ( v42 == 127 )
    v42 = -1;
  sub_15F2440(v18, v42);
  sub_15F2120(v18, v24);
  v70[0] = sub_1649960(v24);
  v72 = 261;
  v70[1] = v43;
  v71[0] = (__int64)v70;
  sub_164B780(v18, v71);
  v44 = v24;
  v45 = (__int64 *)(v18 + 48);
  sub_164D160(v44, v18, a3, a4, a5, a6, v46, v47, a9, a10);
  v48 = *(_QWORD *)(a2 + 48);
  v71[0] = v48;
  if ( v48 )
  {
    sub_1623A60((__int64)v71, v48, 2);
    if ( v45 == v71 )
    {
      if ( v71[0] )
        sub_161E7C0((__int64)v71, v71[0]);
      goto LABEL_48;
    }
    v51 = *(_QWORD *)(v18 + 48);
    if ( !v51 )
    {
LABEL_60:
      v52 = (unsigned __int8 *)v71[0];
      *(_QWORD *)(v18 + 48) = v71[0];
      if ( v52 )
        sub_1623210((__int64)v71, v52, v18 + 48);
      goto LABEL_48;
    }
LABEL_59:
    sub_161E7C0(v18 + 48, v51);
    goto LABEL_60;
  }
  if ( v45 != v71 )
  {
    v51 = *(_QWORD *)(v18 + 48);
    if ( v51 )
      goto LABEL_59;
  }
LABEL_48:
  v71[0] = a2;
  sub_1A062A0(a1 + 64, v71);
  *(_BYTE *)(a1 + 752) = 1;
  if ( v67 == v74 )
  {
    v55 = v75;
    if ( v75 )
    {
      v56 = 32LL * *(_QWORD *)(v75 - 8);
      v57 = v75 + v56;
      if ( v75 != v75 + v56 )
      {
        do
        {
          v57 -= 32;
          if ( v67 == *(void **)(v57 + 8) )
          {
            v58 = *(_QWORD *)(v57 + 16);
            if ( v58 )
            {
              v59 = 32LL * *(_QWORD *)(v58 - 8);
              v60 = v58 + v59;
              while ( v58 != v60 )
              {
                v60 -= 32;
                if ( v67 == *(void **)(v60 + 8) )
                {
                  v61 = *(_QWORD *)(v60 + 16);
                  if ( v61 )
                  {
                    v62 = 32LL * *(_QWORD *)(v61 - 8);
                    v63 = v61 + v62;
                    if ( v61 != v61 + v62 )
                    {
                      do
                      {
                        v66 = v61;
                        v69 = v63 - 32;
                        sub_127D120((_QWORD *)(v63 - 24));
                        v63 = v69;
                        v61 = v66;
                      }
                      while ( v66 != v69 );
                    }
                    j_j_j___libc_free_0_0(v61 - 8);
                  }
                }
                else
                {
                  sub_1698460(v60 + 8);
                }
              }
              j_j_j___libc_free_0_0(v58 - 8);
            }
          }
          else
          {
            sub_1698460(v57 + 8);
          }
        }
        while ( v55 != v57 );
      }
      j_j_j___libc_free_0_0(v55 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v74);
  }
  return v18;
}
