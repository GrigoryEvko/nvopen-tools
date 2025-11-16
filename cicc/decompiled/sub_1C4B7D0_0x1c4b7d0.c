// Function: sub_1C4B7D0
// Address: 0x1c4b7d0
//
void __fastcall sub_1C4B7D0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 *v12; // r15
  unsigned __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  unsigned __int64 v18; // rbx
  unsigned __int8 *v19; // rsi
  unsigned __int64 v20; // r13
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  __int64 *v23; // r12
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int8 *v26; // r8
  __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 *v31; // r13
  __int64 v32; // rsi
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 *v35; // rax
  _QWORD *v36; // rsi
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // rax
  int v40; // r15d
  _QWORD *v41; // r12
  unsigned __int64 *v42; // r15
  __int64 v43; // rax
  unsigned __int64 v44; // rsi
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  __int64 v47; // rdi
  __int64 v48; // rax
  _QWORD *v49; // r13
  __int64 v50; // rsi
  unsigned __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rdx
  _QWORD *v54; // r13
  __int64 v55; // [rsp+0h] [rbp-E0h]
  __int64 v56; // [rsp+8h] [rbp-D8h]
  char v57; // [rsp+17h] [rbp-C9h]
  __int64 v58; // [rsp+18h] [rbp-C8h]
  const char *v59; // [rsp+20h] [rbp-C0h] BYREF
  char v60; // [rsp+30h] [rbp-B0h]
  char v61; // [rsp+31h] [rbp-AFh]
  unsigned __int8 *v62[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v63; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v64; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v65; // [rsp+68h] [rbp-78h]
  unsigned __int64 *v66; // [rsp+70h] [rbp-70h]
  __int64 v67; // [rsp+78h] [rbp-68h]
  __int64 v68; // [rsp+80h] [rbp-60h]
  int v69; // [rsp+88h] [rbp-58h]
  __int64 v70; // [rsp+90h] [rbp-50h]
  __int64 v71; // [rsp+98h] [rbp-48h]

  v10 = *(unsigned int *)(a1 + 8);
  v58 = 0;
  v55 = 8 * v10;
  if ( !(_DWORD)v10 )
    return;
  do
  {
    v12 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + v58) + 40LL) & 0xFFFFFFFFFFFFFFF8LL);
LABEL_3:
    if ( !v12 )
    {
LABEL_79:
      v9 = sub_16498A0(0);
      v64 = 0;
      v66 = 0;
      v67 = v9;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v65 = 0;
      BUG();
    }
    while ( 1 )
    {
      v13 = v12 - 3;
      v14 = sub_16498A0((__int64)(v12 - 3));
      v68 = 0;
      v67 = v14;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v16 = v12[2];
      v17 = (unsigned __int8 *)v12[3];
      v64 = 0;
      v65 = v16;
      v66 = v12;
      v62[0] = v17;
      if ( v17 )
      {
        sub_1623A60((__int64)v62, (__int64)v17, 2);
        if ( v64 )
          sub_161E7C0((__int64)&v64, (__int64)v64);
        v17 = v62[0];
        v64 = v62[0];
        if ( v62[0] )
          sub_1623210((__int64)v62, v62[0], (__int64)&v64);
      }
      if ( v12 == *(unsigned __int64 **)(*(_QWORD *)(*(_QWORD *)a1 + v58) + 48LL) )
        break;
      v18 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *((_BYTE *)v12 - 8) == 38 )
      {
        v57 = 0;
LABEL_17:
        if ( (*((_BYTE *)v12 - 1) & 0x40) != 0 )
        {
          v20 = *(_QWORD *)(*(v12 - 4) + 24);
          v21 = *(_BYTE *)(v20 + 16);
          if ( v21 <= 0x10u )
            goto LABEL_19;
LABEL_27:
          if ( v21 > 0x17u )
          {
            v27 = *(_QWORD *)(v20 + 8);
            if ( v27 )
            {
              if ( !*(_QWORD *)(v27 + 8) && v21 == 40 )
              {
                if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
                {
                  v28 = *(_QWORD **)(v20 - 8);
                  v29 = *v28;
                  if ( *(_BYTE *)(*v28 + 16LL) > 0x10u )
                    goto LABEL_66;
LABEL_33:
                  v30 = sub_15A2BF0((__int64 *)v29, (__int64)v17, v27, v58, *(double *)a2.m128_u64, a3, a4);
                  if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
                    v31 = *(__int64 **)(v20 - 8);
                  else
                    v31 = (__int64 *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
                  if ( *v31 )
                  {
                    v32 = v31[1];
                    v33 = v31[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v33 = v32;
                    if ( v32 )
                      *(_QWORD *)(v32 + 16) = *(_QWORD *)(v32 + 16) & 3LL | v33;
                  }
                  *v31 = v30;
                  if ( v30 )
                  {
                    v34 = *(_QWORD *)(v30 + 8);
                    v31[1] = v34;
                    if ( v34 )
                      *(_QWORD *)(v34 + 16) = (unsigned __int64)(v31 + 1) | *(_QWORD *)(v34 + 16) & 3LL;
                    v31[2] = (v30 + 8) | v31[2] & 3;
                    *(_QWORD *)(v30 + 8) = v31;
                  }
LABEL_42:
                  v62[0] = "conv2Add";
                  v63 = 259;
                  if ( (*((_BYTE *)v12 - 1) & 0x40) != 0 )
                    v35 = (__int64 *)*(v12 - 4);
                  else
                    v35 = (__int64 *)&v13[-3 * (*((_DWORD *)v12 - 1) & 0xFFFFFFF)];
                  v36 = sub_156D730((__int64 *)&v64, *v35, v35[3], (__int64)v62, 0);
                  if ( v36 )
                  {
                    sub_164D160((__int64)(v12 - 3), (__int64)v36, a2, a3, a4, a5, v37, v38, a8, a9);
                    sub_15F20C0(v12 - 3);
                  }
                }
                else
                {
                  v27 = 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF);
                  v28 = (_QWORD *)(v20 - v27);
                  v29 = *(_QWORD *)(v20 - v27);
                  if ( *(_BYTE *)(v29 + 16) <= 0x10u )
                    goto LABEL_33;
LABEL_66:
                  v47 = v28[3];
                  if ( *(_BYTE *)(v47 + 16) <= 0x10u )
                  {
                    v48 = sub_15A2BF0((__int64 *)v47, (__int64)v17, v27, v58, *(double *)a2.m128_u64, a3, a4);
                    if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
                      v49 = *(_QWORD **)(v20 - 8);
                    else
                      v49 = (_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
                    if ( v49[3] )
                    {
                      v50 = v49[4];
                      v51 = v49[5] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v51 = v50;
                      if ( v50 )
                        *(_QWORD *)(v50 + 16) = v51 | *(_QWORD *)(v50 + 16) & 3LL;
                    }
                    v49[3] = v48;
                    if ( v48 )
                    {
                      v52 = *(_QWORD *)(v48 + 8);
                      v49[4] = v52;
                      if ( v52 )
                        *(_QWORD *)(v52 + 16) = (unsigned __int64)(v49 + 4) | *(_QWORD *)(v52 + 16) & 3LL;
                      v53 = v49[5];
                      v54 = v49 + 3;
                      v54[2] = (v48 + 8) | v53 & 3;
                      *(_QWORD *)(v48 + 8) = v54;
                    }
                    goto LABEL_42;
                  }
                }
              }
            }
          }
LABEL_55:
          v26 = v64;
          v19 = v64;
        }
        else
        {
          v15 = 24LL * (*((_DWORD *)v12 - 1) & 0xFFFFFFF);
          v20 = v12[v15 / 0xFFFFFFFFFFFFFFF8LL];
          v21 = *(_BYTE *)(v20 + 16);
          if ( v21 > 0x10u )
            goto LABEL_27;
LABEL_19:
          v61 = 1;
          v59 = "conv2Add";
          v60 = 3;
          v22 = sub_15A2BF0((__int64 *)v20, (__int64)v17, v15, v58, *(double *)a2.m128_u64, a3, a4);
          if ( (*((_BYTE *)v12 - 1) & 0x40) != 0 )
            v23 = (__int64 *)*(v12 - 4);
          else
            v23 = (__int64 *)&v13[-3 * (*((_DWORD *)v12 - 1) & 0xFFFFFFF)];
          v24 = *v23;
          if ( *(_BYTE *)(v24 + 16) > 0x10u
            || *(_BYTE *)(v22 + 16) > 0x10u
            || (v56 = v22,
                v25 = sub_15A2A30((__int64 *)0xC, (__int64 *)v24, v22, 0, 0, *(double *)a2.m128_u64, a3, a4),
                v26 = v64,
                v22 = v56,
                v19 = v64,
                !v25) )
          {
            v63 = 257;
            v39 = sub_15FB440(12, (__int64 *)v24, v22, (__int64)v62, 0);
            v40 = v69;
            v41 = (_QWORD *)v39;
            if ( v68 )
              sub_1625C10(v39, 3, v68);
            sub_15F2440((__int64)v41, v40);
            if ( v65 )
            {
              v42 = v66;
              sub_157E9D0(v65 + 40, (__int64)v41);
              v43 = v41[3];
              v44 = *v42;
              v41[4] = v42;
              v44 &= 0xFFFFFFFFFFFFFFF8LL;
              v41[3] = v44 | v43 & 7;
              *(_QWORD *)(v44 + 8) = v41 + 3;
              *v42 = *v42 & 7 | (unsigned __int64)(v41 + 3);
            }
            sub_164B780((__int64)v41, (__int64 *)&v59);
            if ( !v64 )
            {
              if ( v57 )
                goto LABEL_61;
              v12 = (unsigned __int64 *)v18;
              goto LABEL_3;
            }
            v62[0] = v64;
            sub_1623A60((__int64)v62, (__int64)v64, 2);
            v45 = v41[6];
            if ( v45 )
              sub_161E7C0((__int64)(v41 + 6), v45);
            v46 = v62[0];
            v41[6] = v62[0];
            if ( v46 )
              sub_1623210((__int64)v62, v46, (__int64)(v41 + 6));
            goto LABEL_55;
          }
        }
        if ( v57 )
          goto LABEL_59;
        v12 = (unsigned __int64 *)v18;
        goto LABEL_12;
      }
      v19 = v64;
      v12 = (unsigned __int64 *)(*v12 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_12:
      if ( !v19 )
        goto LABEL_3;
      sub_161E7C0((__int64)&v64, (__int64)v19);
      if ( !v12 )
        goto LABEL_79;
    }
    if ( *((_BYTE *)v12 - 8) == 38 )
    {
      v57 = 1;
      v18 = (unsigned __int64)v12;
      goto LABEL_17;
    }
    v26 = v64;
LABEL_59:
    if ( v26 )
      sub_161E7C0((__int64)&v64, (__int64)v26);
LABEL_61:
    v58 += 8;
  }
  while ( v58 != v55 );
}
