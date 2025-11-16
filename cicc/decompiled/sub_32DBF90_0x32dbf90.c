// Function: sub_32DBF90
// Address: 0x32dbf90
//
__int64 __fastcall sub_32DBF90(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // r13
  unsigned int v7; // ecx
  __int64 v8; // rbx
  __m128i v9; // xmm1
  __int64 v10; // rax
  unsigned __int16 v11; // bx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // r9d
  __int64 v23; // rax
  char v24; // dl
  unsigned int v25; // r10d
  __int64 v26; // rsi
  __int64 v27; // rbx
  int v28; // eax
  char v29; // al
  unsigned int v30; // esi
  __int64 v31; // rax
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned int v35; // [rsp+Ch] [rbp-B4h]
  bool v36; // [rsp+1Bh] [rbp-A5h]
  int v37; // [rsp+1Ch] [rbp-A4h]
  bool v38; // [rsp+1Ch] [rbp-A4h]
  unsigned int v39; // [rsp+1Ch] [rbp-A4h]
  unsigned int v40; // [rsp+1Ch] [rbp-A4h]
  __m128i v41; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+30h] [rbp-90h]
  unsigned int v43; // [rsp+38h] [rbp-88h]
  unsigned int v44; // [rsp+3Ch] [rbp-84h]
  __m128i v45; // [rsp+40h] [rbp-80h] BYREF
  int v46; // [rsp+50h] [rbp-70h] BYREF
  __int64 v47; // [rsp+58h] [rbp-68h]
  __int64 v48; // [rsp+60h] [rbp-60h] BYREF
  int v49; // [rsp+68h] [rbp-58h]
  _OWORD v50[5]; // [rsp+70h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128((const __m128i *)v4);
  v6 = *(_QWORD *)v4;
  v7 = *(_DWORD *)(v4 + 8);
  v8 = *(_QWORD *)(v4 + 40);
  v9 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  LODWORD(v4) = *(_DWORD *)(v4 + 48);
  v45 = v5;
  v42 = v8;
  v37 = v4;
  v10 = *(_QWORD *)(v6 + 48) + 16LL * v7;
  v43 = v7;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v41 = v9;
  v47 = v12;
  LODWORD(v12) = *(_DWORD *)(a2 + 24);
  v13 = *(_QWORD *)(a2 + 80);
  LOWORD(v46) = v11;
  v44 = v12;
  v48 = v13;
  if ( v13 )
    sub_B96E90((__int64)&v48, v13, 1);
  v14 = *a1;
  v15 = _mm_load_si128(&v45);
  v16 = _mm_load_si128(&v41);
  v49 = *(_DWORD *)(a2 + 72);
  v50[0] = v15;
  v50[1] = v16;
  v17 = sub_3402EA0(v14, v44, (unsigned int)&v48, v46, v47, 0, (__int64)v50, 2);
  if ( v17 )
    goto LABEL_4;
  v38 = v37 == v43 && v42 == v6;
  if ( v38 )
  {
    v17 = v45.m128i_i64[0];
    goto LABEL_4;
  }
  if ( (unsigned __int8)sub_33E2390(*a1, v45.m128i_i64[0], v45.m128i_i64[1], 1)
    && !(unsigned __int8)sub_33E2390(*a1, v41.m128i_i64[0], v41.m128i_i64[1], 1) )
  {
    v31 = sub_3406EB0(*a1, v44, (unsigned int)&v48, v46, v47, v32, *(_OWORD *)&v41, *(_OWORD *)&v45);
    goto LABEL_39;
  }
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0xD3u )
      goto LABEL_12;
  }
  else if ( !sub_30070B0((__int64)&v46) )
  {
    v17 = sub_3286810(a1, v44, (__int64)&v48, v45.m128i_i64[0], v45.m128i_i64[1], *(_DWORD *)(a2 + 28), *(_OWORD *)&v41);
    if ( v17 )
      goto LABEL_4;
    goto LABEL_28;
  }
  v17 = sub_3295970(a1, a2, (__int64)&v48, v20, v21);
  if ( v17 )
    goto LABEL_4;
LABEL_12:
  v17 = sub_3286810(a1, v44, (__int64)&v48, v45.m128i_i64[0], v45.m128i_i64[1], *(_DWORD *)(a2 + 28), *(_OWORD *)&v41);
  if ( v17 )
    goto LABEL_4;
  v23 = a1[1];
  if ( v11 == 1 || v11 && *(_QWORD *)(v23 + 8LL * v11 + 112) )
  {
    if ( v44 <= 0x1F3 )
    {
      v24 = *(_BYTE *)(v44 + 500LL * v11 + v23 + 6414);
      if ( v44 == 182 )
      {
        v28 = *(_DWORD *)(v6 + 24);
        v36 = v24 != 0;
        if ( v28 == 181 )
          goto LABEL_46;
        if ( !v24 )
        {
          v25 = 2;
LABEL_51:
          v40 = v25;
          v17 = sub_3286970(
                  v45.m128i_i64[0],
                  v45.m128i_i32[2],
                  v41.m128i_i64[0],
                  v41.m128i_i64[1],
                  v6,
                  v43,
                  v41.m128i_i64[0],
                  v41.m128i_i64[1],
                  12,
                  *a1);
          v25 = v40;
          if ( !v17 )
          {
LABEL_19:
            v26 = *(_QWORD *)(a2 + 80);
            *(_QWORD *)&v50[0] = v26;
            if ( !v26 )
            {
              DWORD2(v50[0]) = *(_DWORD *)(a2 + 72);
              goto LABEL_22;
            }
            goto LABEL_20;
          }
LABEL_4:
          v18 = v17;
          goto LABEL_5;
        }
LABEL_30:
        v36 = 1;
        if ( v28 == 51 )
          goto LABEL_31;
        goto LABEL_47;
      }
      if ( !v24 )
      {
        v25 = v44 - 180;
        if ( v44 - 180 > 1 )
          goto LABEL_43;
LABEL_18:
        v39 = v25;
        v17 = sub_3283940(
                v45.m128i_i64[0],
                v45.m128i_u32[2],
                v41.m128i_i64[0],
                v41.m128i_i64[1],
                v6,
                v43,
                v41.m128i_i64[0],
                v41.m128i_i64[1],
                2 * (unsigned int)(v44 == 180) + 18,
                *a1);
        v25 = v39;
        if ( !v17 )
          goto LABEL_19;
        goto LABEL_4;
      }
    }
LABEL_53:
    v28 = *(_DWORD *)(v6 + 24);
    goto LABEL_30;
  }
LABEL_28:
  if ( v44 != 182 )
    goto LABEL_53;
  v28 = *(_DWORD *)(v6 + 24);
  if ( v28 != 181 )
    goto LABEL_30;
  v36 = 1;
LABEL_46:
  v38 = 1;
LABEL_47:
  if ( !(unsigned __int8)sub_33DD2A0(*a1, v45.m128i_i64[0], v45.m128i_i64[1], 0) )
  {
    v25 = v44 - 180;
    goto LABEL_49;
  }
LABEL_31:
  v25 = v44 - 180;
  if ( *(_DWORD *)(v42 + 24) == 51
    || (v35 = v44 - 180, v29 = sub_33DD2A0(*a1, v41.m128i_i64[0], v41.m128i_i64[1], 0), v25 = v35, v29) )
  {
    if ( v25 > 3 )
LABEL_70:
      BUG();
    v30 = dword_44D8240[v25];
    if ( v38 && v36
      || ((v33 = a1[1], v11 == 1) || v11 && *(_QWORD *)(v33 + 8LL * v11 + 112))
      && v30 <= 0x1F3
      && !*(_BYTE *)(v30 + 500LL * v11 + v33 + 6414) )
    {
      v31 = sub_3406EB0(*a1, v30, (unsigned int)&v48, v46, v47, v22, *(_OWORD *)&v45, *(_OWORD *)&v41);
LABEL_39:
      v18 = v31;
      goto LABEL_5;
    }
  }
LABEL_49:
  if ( v25 <= 1 )
    goto LABEL_18;
  if ( v44 == 182 )
    goto LABEL_51;
LABEL_43:
  v26 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v50[0] = v26;
  if ( v26 )
  {
LABEL_20:
    v45.m128i_i32[0] = v25;
    sub_B96E90((__int64)v50, v26, 1);
    v25 = v45.m128i_i32[0];
  }
  DWORD2(v50[0]) = *(_DWORD *)(a2 + 72);
  if ( v25 > 3 )
    goto LABEL_70;
LABEL_22:
  v27 = sub_328C120(a1, dword_44D8270[v25], v44, (int)v50, v46, v47, v6, v42, 0);
  if ( *(_QWORD *)&v50[0] )
    sub_B91220((__int64)v50, *(__int64 *)&v50[0]);
  if ( v27 )
  {
    v18 = v27;
  }
  else
  {
    v34 = (__int64)a1;
    v18 = 0;
    if ( (unsigned __int8)sub_32D0FE0(v34, a2, 0) )
      v18 = a2;
  }
LABEL_5:
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v18;
}
