// Function: sub_70A7F0
// Address: 0x70a7f0
//
void __fastcall sub_70A7F0(
        __int64 a1,
        __int64 a2,
        int a3,
        unsigned __int8 a4,
        __m128i *a5,
        int a6,
        int *a7,
        _DWORD *a8)
{
  int v10; // ebx
  __int64 v11; // r12
  int v12; // r15d
  int v13; // ecx
  int v14; // r14d
  int v15; // eax
  __int64 v16; // rdi
  int v17; // r15d
  int v18; // edx
  char v19; // r8
  int v20; // r14d
  int v21; // eax
  int v22; // ecx
  char v23; // r8
  __int64 v24; // rsi
  char v25; // si
  int v26; // r14d
  int v27; // edx
  int v28; // eax
  int v29; // r8d
  int v30; // edx
  _BOOL4 v31; // eax
  bool v32; // cf
  int v33; // eax
  int v34; // eax
  int v35; // ebx
  int v36; // r14d
  int v37; // eax
  bool v38; // cl
  int v39; // edi
  __m128i *v40; // rsi
  unsigned int v41; // ecx
  int v42; // edi
  int v43; // edx
  __m128i *v44; // rcx
  unsigned int v45; // esi
  int v46; // edx
  unsigned int *v47; // rdx
  unsigned int v48; // r8d
  int *v49; // rdx
  unsigned int v50; // edi
  __m128i *v51; // rsi
  __int32 v52; // edi
  __int32 v53; // ecx
  unsigned int *v54; // rdx
  unsigned int v55; // ecx
  char v56; // [rsp+Ch] [rbp-74h]
  int v57; // [rsp+Ch] [rbp-74h]
  int v58; // [rsp+10h] [rbp-70h]
  int v59; // [rsp+10h] [rbp-70h]
  int v60; // [rsp+10h] [rbp-70h]
  char v61; // [rsp+10h] [rbp-70h]
  int v63; // [rsp+18h] [rbp-68h]
  int v64; // [rsp+1Ch] [rbp-64h]
  char v66; // [rsp+2Bh] [rbp-55h]
  int v67; // [rsp+2Ch] [rbp-54h]
  int v68; // [rsp+30h] [rbp-50h]
  char v69; // [rsp+37h] [rbp-49h]
  __int64 v70; // [rsp+38h] [rbp-48h] BYREF
  __m128i v71; // [rsp+40h] [rbp-40h] BYREF

  v70 = a2;
  *a7 = 0;
  v66 = a4;
  v68 = dword_4F07890;
  if ( dword_4F07890 )
  {
    v69 = a4 - 5;
    if ( (unsigned __int8)(a4 - 5) <= 1u )
    {
      v69 = -1;
      v10 = 4;
      v68 = 0;
      v66 = 4;
    }
    else
    {
      v68 = 0;
      v10 = a4;
    }
  }
  else if ( a4 == 6 )
  {
    v25 = a4;
    v69 = 2 * (unk_4F06930 == 106) + 1;
    v10 = 2 * (unk_4F06930 == 106) + 6;
    v68 = unk_4F06930 == 106;
    if ( unk_4F06930 == 106 )
      v25 = 8;
    v66 = v25;
  }
  else
  {
    v10 = a4;
    v69 = a4 - 5;
  }
  v11 = v10;
  v12 = dword_4D04120[v10];
  v63 = sub_70A480((unsigned int *)a1, 0);
  if ( !v63 )
  {
    v70 = 0;
    goto LABEL_6;
  }
  while ( *(int *)a1 >= 0 )
  {
    sub_70A210((int *)a1, 1);
    --v70;
  }
  sub_70A300((int *)a1, &v70, v12, 0, 0, a8);
  if ( (unsigned __int8)v69 > 1u || !dword_4F07880 )
    sub_70A210((int *)a1, 1);
  --v70;
  if ( !a6 )
  {
LABEL_6:
    v13 = dword_4F07890;
    if ( dword_4F07890 )
      goto LABEL_7;
LABEL_26:
    v57 = v13;
    v26 = dword_4D04060[v10];
    v27 = v26 - 1;
    v20 = v26 - 2;
    v59 = v27;
    v67 = dword_4D04120[v10];
    v64 = dword_4D04020[v10] - 1;
    v28 = sub_70A480((unsigned int *)a1, 0);
    v16 = v70;
    v24 = v20;
    v17 = v28;
    v18 = v59;
    if ( v70 >= v59 )
    {
      v23 = v69;
LABEL_34:
      v31 = 1;
      if ( (unsigned __int8)v23 > 1u )
        goto LABEL_37;
      v30 = dword_4F07880;
      goto LABEL_36;
    }
    v21 = v59 - v70;
    v22 = v57;
    v29 = v17 + v59 - v70;
    if ( (unsigned __int8)v69 <= 1u )
    {
      v30 = dword_4F07880;
      if ( dword_4F07880 )
      {
        if ( v67 >= v29 )
        {
          v23 = v69;
          goto LABEL_12;
        }
      }
      else if ( v67 > v29 )
      {
        v19 = v69;
        goto LABEL_11;
      }
LABEL_36:
      v31 = v30 == 0;
      goto LABEL_37;
    }
    v19 = v69;
    goto LABEL_10;
  }
  v13 = dword_4F07890;
  *a7 = 1;
  if ( !v13 )
    goto LABEL_26;
LABEL_7:
  if ( (unsigned __int8)v69 > 1u )
  {
    v14 = dword_4D04060[v10];
    v67 = dword_4D04120[v10];
    v64 = dword_4D04020[v10] - 1;
    v15 = sub_70A480((unsigned int *)a1, 0);
    v16 = v70;
    v17 = v15;
    v18 = v14 - 1;
    if ( v14 - 1 <= v70 )
    {
      v20 = v14 - 2;
LABEL_63:
      v24 = v20;
      v31 = 1;
      goto LABEL_37;
    }
    v19 = v69;
    v20 = v14 - 2;
LABEL_10:
    v21 = v18 - v16;
    if ( v18 - (int)v16 + v17 < v67 )
    {
LABEL_11:
      v56 = v19;
      v58 = v21;
      sub_70A250((int *)a1, 1);
      *(_DWORD *)a1 |= 0x80000000;
      v21 = v58;
      v22 = 1;
      v23 = v56;
LABEL_12:
      if ( v22 < v21 )
      {
        v61 = v23;
        sub_70A250((int *)a1, v21 - v22);
        v23 = v61;
      }
      v16 = v20;
      v70 = v20;
      v24 = v20;
      goto LABEL_34;
    }
    goto LABEL_63;
  }
  v36 = unk_4D04070;
  v60 = unk_4D04070 - 1;
  v67 = dword_4D04120[4];
  v64 = dword_4D04020[4] - 1;
  v37 = sub_70A480((unsigned int *)a1, 0);
  v16 = v70;
  v17 = v37;
  v18 = v60;
  if ( v70 < v60 )
  {
    v19 = -1;
    LOBYTE(v10) = 4;
    v20 = v36 - 2;
    goto LABEL_10;
  }
  v31 = 1;
  LOBYTE(v10) = 4;
  v24 = v36 - 2;
LABEL_37:
  if ( v17 + v31 > v67 )
    *a8 = 1;
  if ( v16 < v24 || v16 > v64 )
  {
    if ( HIDWORD(qword_4F077B4) )
      sub_70A170(a5, v10);
    *a7 = 1;
  }
  else
  {
    if ( *a7 )
      goto LABEL_43;
    v32 = unk_4F07580 == 0;
    *a5 = 0;
    v33 = v32 ? 1 : -1;
    if ( !v63 )
      goto LABEL_43;
    if ( v66 != 2 )
    {
      v38 = v66 != 14 && (unsigned __int8)v66 > 8u;
      if ( !v38 )
      {
        if ( (unsigned __int8)(v66 - 3) <= 1u )
        {
LABEL_71:
          v40 = &v71;
          if ( unk_4F07580 )
            v40 = (__m128i *)((char *)v71.m128i_i64 + 4);
          v41 = *(_DWORD *)a1;
          v42 = (((_DWORD)v16 + 1023) << 20) | (*(_DWORD *)a1 >> 12);
          if ( a3 )
            v42 |= 0x80000000;
          v43 = *(_DWORD *)(a1 + 4) >> 12;
          v40->m128i_i32[0] = v42;
          v40->m128i_i32[v33] = (v41 << 20) | v43;
          a5->m128i_i64[0] = v71.m128i_i64[0];
          goto LABEL_43;
        }
        if ( (unsigned __int8)v69 > 1u )
        {
LABEL_59:
          if ( v66 == 7 )
          {
            if ( unk_4F06924 != 64 )
              goto LABEL_61;
            goto LABEL_92;
          }
          if ( v66 == 8 )
          {
            if ( unk_4F06918 == 113 )
              goto LABEL_83;
            goto LABEL_61;
          }
LABEL_82:
          if ( v38 )
          {
LABEL_83:
            v44 = (__m128i *)((char *)&v71.m128i_u64[1] + 4);
            if ( !unk_4F07580 )
              v44 = &v71;
            v45 = *(_DWORD *)a1;
            v46 = (((_DWORD)v16 + 0x3FFF) << 16) | HIWORD(*(_DWORD *)a1);
            if ( a3 )
              v46 |= 0x80000000;
            v44->m128i_i32[0] = v46;
            v47 = (unsigned int *)v44 + v33;
            v48 = *(_DWORD *)(a1 + 4);
            *v47 = HIWORD(v48) | (v45 << 16);
            v49 = (int *)&v47[v33];
            v50 = *(_DWORD *)(a1 + 8);
            *v49 = HIWORD(v50) | (v48 << 16);
            v49[v33] = *(unsigned __int16 *)(a1 + 14) | (v50 << 16);
            *a5 = _mm_load_si128(&v71);
            goto LABEL_43;
          }
LABEL_61:
          sub_721090(v16);
        }
LABEL_77:
        if ( !dword_4F07890 && (!v38 || qword_4D040A0[v11] != 8) )
        {
          if ( unk_4F06930 == 64 )
          {
LABEL_92:
            v51 = (__m128i *)&v71.m128i_u64[1];
            if ( !unk_4F07580 )
              v51 = &v71;
            v52 = v16 + 0x3FFF;
            v53 = v52;
            v54 = (unsigned int *)v51 + v33;
            if ( a3 )
            {
              BYTE1(v53) = BYTE1(v52) | 0x80;
              v52 = v53;
            }
            v55 = *(_DWORD *)a1;
            v51->m128i_i32[0] = v52;
            *v54 = v55;
            v54[v33] = *(_DWORD *)(a1 + 4);
            a5->m128i_i64[0] = v71.m128i_i64[0];
            a5->m128i_i32[2] = v71.m128i_i32[2];
            goto LABEL_43;
          }
          if ( unk_4F06930 == 113 )
            goto LABEL_83;
          goto LABEL_82;
        }
        goto LABEL_71;
      }
      if ( dword_4D04120[v11] > dword_4D04120[2] )
      {
        if ( (unsigned __int8)v69 > 1u )
        {
          if ( qword_4D040A0[v11] != 8 )
            goto LABEL_59;
          goto LABEL_71;
        }
        goto LABEL_77;
      }
    }
    v39 = (*(_DWORD *)a1 >> 9) | (((_DWORD)v16 + 127) << 23);
    if ( a3 )
      v39 |= 0x80000000;
    a5->m128i_i32[0] = v39;
  }
LABEL_43:
  v34 = *(_DWORD *)(a1 + 16);
  if ( v34 )
    *a8 = v34;
  if ( v68 )
  {
    v35 = *a7;
    sub_709EF0(a5, 8u, a5, 6u, a7, &v71);
    if ( v35 )
      *a7 = 1;
  }
}
