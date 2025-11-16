// Function: sub_2A5D380
// Address: 0x2a5d380
//
__int64 __fastcall sub_2A5D380(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rdx
  __int64 v5; // rdx
  unsigned __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rbx
  __int64 v12; // rbx
  int *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // r14
  unsigned __int64 v21; // rbx
  _QWORD *v22; // rcx
  unsigned __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rcx
  __int64 v27; // rcx
  unsigned __int128 v28; // rdi
  int *v29; // rax
  int *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r13
  int *v33; // r15
  int *v34; // rdi
  int *v35; // rax
  _BYTE *v37; // rax
  _BYTE *v38; // rsi
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 **v41; // rbx
  __int64 v42; // rcx
  __int64 *v43; // rax
  __int64 *v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rsi
  _BYTE *v47; // rax
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdx
  _QWORD *v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rbx
  __int64 v56; // [rsp+20h] [rbp-E0h]
  char *v57; // [rsp+28h] [rbp-D8h]
  __int64 v58; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v59; // [rsp+48h] [rbp-B8h]
  __int64 *v60; // [rsp+50h] [rbp-B0h]
  __int64 *v62; // [rsp+60h] [rbp-A0h]
  _QWORD *v63; // [rsp+68h] [rbp-98h]
  __int64 v64; // [rsp+70h] [rbp-90h]
  __int64 *v65; // [rsp+78h] [rbp-88h]
  __m128i v66; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v67; // [rsp+90h] [rbp-70h]
  __int64 v68; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-58h] BYREF
  unsigned __int64 v70; // [rsp+B0h] [rbp-50h]
  int *v71; // [rsp+B8h] [rbp-48h]
  __int64 *v72; // [rsp+C0h] [rbp-40h]
  __int64 v73; // [rsp+C8h] [rbp-38h]

  v56 = a4;
  if ( a4 == a3
    || (v4 = (_QWORD *)a2[1], *(_QWORD *)(*v4 + 80 * a3 + 32) == *(_QWORD *)(*v4 + 80 * a3 + 40)) && a4 == -1 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_OWORD *)a1 = 0;
    return a1;
  }
  v5 = v4[1] - *v4;
  if ( v5 < 0 )
    goto LABEL_76;
  if ( !(0xCCCCCCCCCCCCCCCDLL * (v5 >> 4)) )
  {
    v59 = 0;
    goto LABEL_33;
  }
  v6 = 0xCCCCCCCCCCCCCCDLL * (v5 >> 4);
  v7 = (_QWORD *)sub_22077B0(v6 * 8);
  v8 = &v7[v6];
  v59 = (unsigned __int64)v7;
  if ( v7 == v8 )
  {
    v9 = *(_QWORD *)(a2[1] + 8LL) - *(_QWORD *)a2[1];
    v10 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 4);
  }
  else
  {
    do
      *v7++ = 0x4000000000000LL;
    while ( v7 != v8 );
    v9 = *(_QWORD *)(a2[1] + 8LL) - *(_QWORD *)a2[1];
    v10 = 0xCCCCCCCCCCCCCCCDLL * (v9 >> 4);
  }
  if ( v9 < 0 )
LABEL_76:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v10 )
  {
    v11 = 8 * v10;
    v57 = (char *)sub_22077B0(v11);
    if ( v57 != &v57[v11] )
      memset(v57, 0, v11);
    goto LABEL_13;
  }
LABEL_33:
  v57 = 0;
LABEL_13:
  LODWORD(v69) = 0;
  *(_QWORD *)(v59 + 8 * a3) = 0;
  v66.m128i_i64[1] = a3;
  v70 = 0;
  v71 = (int *)&v69;
  v72 = &v69;
  v73 = 0;
  v66.m128i_i64[0] = 0;
  sub_2A5C740(&v68, &v66);
  if ( v73 )
  {
    do
    {
      v12 = *((_QWORD *)v71 + 5);
      v13 = sub_220F330(v71, &v69);
      j_j___libc_free_0((unsigned __int64)v13);
      v14 = --v73;
      if ( v56 == v12 )
        break;
      v15 = (__int64 *)a2[1];
      v16 = *v15;
      v17 = *v15 + 80 * v12;
      v18 = *(__int64 **)(v17 + 40);
      v19 = *(__int64 **)(v17 + 32);
      v60 = v18;
      if ( v18 == v19 )
      {
        if ( v56 == -1 )
          goto LABEL_58;
      }
      else
      {
        v65 = v19;
        v63 = (_QWORD *)(v59 + 8 * v12);
        do
        {
          v20 = *v65;
          v64 = *(_QWORD *)(*v65 + 8);
          v21 = *(_QWORD *)(*a2 + 56LL);
          if ( !*(_BYTE *)(*v65 + 25) )
          {
            v22 = (_QWORD *)a2[1];
            v23 = 0x999999999999999ALL * ((__int64)(v22[1] - *v22) >> 4) + 2;
            v24 = *v22 + 80LL * v22[6];
            v25 = v21 / v23;
            if ( *(_QWORD *)(v24 + 24) <= v25 )
              v25 = *(_QWORD *)(v24 + 24);
            v26 = *(_QWORD *)(v20 + 32);
            if ( v25 < 0x2710 )
              v25 = 10000;
            if ( v26 )
              v21 = v25 / v26 + v25;
            else
              v21 = v23 * v25;
          }
          v62 = (__int64 *)(v59 + 8 * v64);
          v27 = v21 + *v63;
          if ( *v62 > v27 )
          {
            v66.m128i_i64[0] = *v62;
            *((_QWORD *)&v28 + 1) = &v66;
            *(_QWORD *)&v28 = &v68;
            v58 = v27;
            v66.m128i_i64[1] = v64;
            v29 = (int *)sub_2A5D2A0(v28);
            v31 = v58;
            v32 = (__int64)v29;
            if ( v71 == v29 && &v69 == (__int64 *)v30 )
            {
              sub_2A5B960(v70);
              v71 = (int *)&v69;
              v70 = 0;
              v53 = *v63 + v21;
              v72 = &v69;
              v73 = 0;
              v31 = v53;
            }
            else
            {
              v33 = v30;
              if ( v30 != v29 )
              {
                do
                {
                  v34 = (int *)v32;
                  v32 = sub_220EF30(v32);
                  v35 = sub_220F330(v34, &v69);
                  j_j___libc_free_0((unsigned __int64)v35);
                  --v73;
                }
                while ( v33 != (int *)v32 );
                v31 = *v63 + v21;
              }
            }
            v66.m128i_i64[0] = v31;
            *v62 = v31;
            *(_QWORD *)&v57[8 * v64] = v20;
            v66.m128i_i64[1] = v64;
            sub_2A5C740(&v68, &v66);
          }
          ++v65;
        }
        while ( v60 != v65 );
        v14 = v73;
      }
    }
    while ( v14 );
  }
  if ( v56 != -1 )
  {
    v66 = 0u;
    v67 = 0;
    goto LABEL_41;
  }
  v15 = (__int64 *)a2[1];
  v16 = *v15;
LABEL_58:
  v49 = 0xCCCCCCCCCCCCCCCDLL * ((v15[1] - v16) >> 4);
  if ( !v49 )
  {
    v56 = -1;
    goto LABEL_68;
  }
  v50 = (_QWORD *)(v16 + 32);
  v51 = 0;
  v52 = -1;
  do
  {
    while ( 1 )
    {
      if ( v50[1] != *v50 || !*(_QWORD *)&v57[8 * v51] )
        goto LABEL_60;
      if ( v52 != -1 )
        break;
      v52 = v51;
LABEL_60:
      ++v51;
      v50 += 10;
      if ( v51 == v49 )
        goto LABEL_67;
    }
    if ( *(_QWORD *)(v59 + 8 * v52) > *(_QWORD *)(v59 + 8 * v51) )
      v52 = v51;
    ++v51;
    v50 += 10;
  }
  while ( v51 != v49 );
LABEL_67:
  v56 = v52;
LABEL_68:
  v66 = 0u;
  v67 = 0;
  if ( v56 == a3 )
  {
    v47 = 0;
    v38 = 0;
    v42 = 0;
    goto LABEL_53;
  }
LABEL_41:
  v37 = 0;
  v38 = 0;
  v39 = v56;
  while ( 2 )
  {
    v41 = (__int64 **)&v57[8 * v39];
    if ( v37 == v38 )
    {
      sub_2A5C8B0((__int64)&v66, v38, &v57[8 * v39]);
      v38 = (_BYTE *)v66.m128i_i64[1];
      v39 = **v41;
      if ( v39 == a3 )
        break;
      goto LABEL_45;
    }
    v40 = *v41;
    if ( v38 )
    {
      *(_QWORD *)v38 = v40;
      v38 = (_BYTE *)v66.m128i_i64[1];
    }
    v38 += 8;
    v66.m128i_i64[1] = (__int64)v38;
    v39 = *v40;
    if ( *v40 != a3 )
    {
LABEL_45:
      v37 = v67;
      continue;
    }
    break;
  }
  v42 = v66.m128i_i64[0];
  if ( (_BYTE *)v66.m128i_i64[0] != v38 )
  {
    v43 = (__int64 *)(v38 - 8);
    v44 = (__int64 *)v66.m128i_i64[0];
    if ( v66.m128i_i64[0] < (unsigned __int64)(v38 - 8) )
    {
      do
      {
        v45 = *v44;
        v46 = *v43;
        ++v44;
        --v43;
        *(v44 - 1) = v46;
        v43[1] = v45;
      }
      while ( v43 > v44 );
      v38 = (_BYTE *)v66.m128i_i64[1];
      v42 = v66.m128i_i64[0];
    }
  }
  v47 = v67;
LABEL_53:
  v48 = v70;
  *(_QWORD *)(a1 + 16) = v47;
  *(_QWORD *)a1 = v42;
  *(_QWORD *)(a1 + 8) = v38;
  sub_2A5B960(v48);
  if ( v57 )
    j_j___libc_free_0((unsigned __int64)v57);
  if ( v59 )
    j_j___libc_free_0(v59);
  return a1;
}
