// Function: sub_1A62BF0
// Address: 0x1a62bf0
//
__int64 __fastcall sub_1A62BF0(int a1, char a2, char a3, char a4, char a5, char a6, char a7, __m128i *a8)
{
  __int64 v11; // rcx
  __m128i v12; // xmm1
  void (__fastcall *v13)(__m128i *, __m128i *, __int64); // rdx
  __m128i v14; // xmm0
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r15
  __m128i v18; // xmm0
  __m128i v19; // xmm2
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rax
  _DWORD *v25; // r9
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  _DWORD *v29; // r10
  _DWORD *v30; // r9
  __int64 v31; // rcx
  __int64 v32; // rdx
  int v33; // eax
  unsigned __int64 v34; // rdi
  _QWORD *v35; // rax
  _DWORD *v36; // r9
  __int64 v37; // rcx
  __int64 v38; // rdx
  unsigned __int64 v39; // rdi
  _QWORD *v40; // rax
  _DWORD *v41; // r9
  __int64 v42; // rcx
  __int64 v43; // rdx
  unsigned __int64 v44; // rdi
  _QWORD *v45; // rax
  _DWORD *v46; // r9
  __int64 v47; // rcx
  __int64 v48; // rdx
  unsigned __int64 v49; // rdi
  _QWORD *v50; // rax
  _DWORD *v51; // r9
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rax
  _DWORD *v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v59; // rax
  _DWORD *v60; // r10
  _DWORD *v61; // r9
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  _DWORD *v65; // r10
  _DWORD *v66; // r9
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rax
  _DWORD *v70; // r10
  _DWORD *v71; // r9
  __int64 v72; // rcx
  __int64 v73; // rdx
  __m128i v77; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v78)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-40h]
  __int64 v79; // [rsp+28h] [rbp-38h]

  v11 = v79;
  v12 = _mm_loadu_si128(&v77);
  v13 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a8[1].m128i_i64[0];
  v14 = _mm_loadu_si128(a8);
  a8[1].m128i_i64[0] = 0;
  *a8 = v12;
  v78 = v13;
  v15 = a8[1].m128i_i64[1];
  a8[1].m128i_i64[1] = v11;
  v79 = v15;
  v77 = v14;
  v16 = sub_22077B0(216);
  v17 = v16;
  if ( v16 )
  {
    *(_QWORD *)(v16 + 8) = 0;
    v18 = _mm_loadu_si128(&v77);
    *(_QWORD *)(v16 + 16) = &unk_4FB43CC;
    v19 = _mm_loadu_si128((const __m128i *)(v16 + 184));
    *(_QWORD *)(v16 + 80) = v16 + 64;
    v20 = *(_QWORD *)(v16 + 208);
    *(_QWORD *)(v16 + 88) = v16 + 64;
    *(_QWORD *)(v16 + 128) = v16 + 112;
    *(_QWORD *)(v16 + 136) = v16 + 112;
    *(_QWORD *)v16 = off_49F5528;
    *(_QWORD *)(v16 + 160) = 0x1000000000001LL;
    v77 = v19;
    *(_QWORD *)(v16 + 200) = v78;
    v21 = v79;
    *(__m128i *)(v17 + 184) = v18;
    v79 = v20;
    *(_DWORD *)(v17 + 24) = 3;
    *(_QWORD *)(v17 + 32) = 0;
    *(_QWORD *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 48) = 0;
    *(_DWORD *)(v17 + 64) = 0;
    *(_QWORD *)(v17 + 72) = 0;
    *(_QWORD *)(v17 + 96) = 0;
    *(_DWORD *)(v17 + 112) = 0;
    *(_QWORD *)(v17 + 120) = 0;
    *(_QWORD *)(v17 + 144) = 0;
    *(_BYTE *)(v17 + 152) = 0;
    *(_QWORD *)(v17 + 176) = 0;
    v78 = 0;
    *(_QWORD *)(v17 + 208) = v21;
    v22 = sub_163A1D0();
    sub_1A62500(v22);
    v23 = sub_16D5D50();
    v24 = *(_QWORD **)&dword_4FA0208[2];
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      goto LABEL_85;
    v25 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v26 = v24[2];
        v27 = v24[3];
        if ( v23 <= v24[4] )
          break;
        v24 = (_QWORD *)v24[3];
        if ( !v27 )
          goto LABEL_7;
      }
      v25 = v24;
      v24 = (_QWORD *)v24[2];
    }
    while ( v26 );
LABEL_7:
    if ( v25 == dword_4FA0208 )
      goto LABEL_85;
    if ( v23 < *((_QWORD *)v25 + 4) )
      goto LABEL_85;
    v28 = *((_QWORD *)v25 + 7);
    v29 = v25 + 12;
    if ( !v28 )
      goto LABEL_85;
    v30 = v25 + 12;
    do
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v28 + 16);
        v32 = *(_QWORD *)(v28 + 24);
        if ( *(_DWORD *)(v28 + 32) >= dword_4FB4768 )
          break;
        v28 = *(_QWORD *)(v28 + 24);
        if ( !v32 )
          goto LABEL_14;
      }
      v30 = (_DWORD *)v28;
      v28 = *(_QWORD *)(v28 + 16);
    }
    while ( v31 );
LABEL_14:
    if ( v29 == v30 || dword_4FB4768 < v30[8] || (v33 = dword_4FB4800, !v30[9]) )
LABEL_85:
      v33 = a1;
    *(_DWORD *)(v17 + 160) = v33;
    v34 = sub_16D5D50();
    v35 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v36 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v37 = v35[2];
          v38 = v35[3];
          if ( v34 <= v35[4] )
            break;
          v35 = (_QWORD *)v35[3];
          if ( !v38 )
            goto LABEL_22;
        }
        v36 = v35;
        v35 = (_QWORD *)v35[2];
      }
      while ( v37 );
LABEL_22:
      if ( v36 != dword_4FA0208 && v34 >= *((_QWORD *)v36 + 4) )
      {
        v59 = *((_QWORD *)v36 + 7);
        v60 = v36 + 12;
        if ( v59 )
        {
          v61 = v36 + 12;
          do
          {
            while ( 1 )
            {
              v62 = *(_QWORD *)(v59 + 16);
              v63 = *(_QWORD *)(v59 + 24);
              if ( *(_DWORD *)(v59 + 32) >= dword_4FB44C8 )
                break;
              v59 = *(_QWORD *)(v59 + 24);
              if ( !v63 )
                goto LABEL_63;
            }
            v61 = (_DWORD *)v59;
            v59 = *(_QWORD *)(v59 + 16);
          }
          while ( v62 );
LABEL_63:
          if ( v60 != v61 && dword_4FB44C8 >= v61[8] && v61[9] )
            a2 = byte_4FB4560;
        }
      }
    }
    *(_BYTE *)(v17 + 164) = a2;
    v39 = sub_16D5D50();
    v40 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v41 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v42 = v40[2];
          v43 = v40[3];
          if ( v39 <= v40[4] )
            break;
          v40 = (_QWORD *)v40[3];
          if ( !v43 )
            goto LABEL_29;
        }
        v41 = v40;
        v40 = (_QWORD *)v40[2];
      }
      while ( v42 );
LABEL_29:
      if ( v41 != dword_4FA0208 && v39 >= *((_QWORD *)v41 + 4) )
      {
        v69 = *((_QWORD *)v41 + 7);
        v70 = v41 + 12;
        if ( v69 )
        {
          v71 = v41 + 12;
          do
          {
            while ( 1 )
            {
              v72 = *(_QWORD *)(v69 + 16);
              v73 = *(_QWORD *)(v69 + 24);
              if ( *(_DWORD *)(v69 + 32) >= dword_4FB45A8 )
                break;
              v69 = *(_QWORD *)(v69 + 24);
              if ( !v73 )
                goto LABEL_81;
            }
            v71 = (_DWORD *)v69;
            v69 = *(_QWORD *)(v69 + 16);
          }
          while ( v72 );
LABEL_81:
          if ( v70 != v71 && dword_4FB45A8 >= v71[8] && v71[9] )
            a3 = byte_4FB4640;
        }
      }
    }
    *(_BYTE *)(v17 + 165) = a3;
    v44 = sub_16D5D50();
    v45 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v46 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v47 = v45[2];
          v48 = v45[3];
          if ( v44 <= v45[4] )
            break;
          v45 = (_QWORD *)v45[3];
          if ( !v48 )
            goto LABEL_36;
        }
        v46 = v45;
        v45 = (_QWORD *)v45[2];
      }
      while ( v47 );
LABEL_36:
      if ( v46 != dword_4FA0208 && v44 >= *((_QWORD *)v46 + 4) )
      {
        v64 = *((_QWORD *)v46 + 7);
        v65 = v46 + 12;
        if ( v64 )
        {
          v66 = v46 + 12;
          do
          {
            while ( 1 )
            {
              v67 = *(_QWORD *)(v64 + 16);
              v68 = *(_QWORD *)(v64 + 24);
              if ( *(_DWORD *)(v64 + 32) >= dword_4FB4688 )
                break;
              v64 = *(_QWORD *)(v64 + 24);
              if ( !v68 )
                goto LABEL_72;
            }
            v66 = (_DWORD *)v64;
            v64 = *(_QWORD *)(v64 + 16);
          }
          while ( v67 );
LABEL_72:
          if ( v65 != v66 && dword_4FB4688 >= v66[8] && v66[9] )
            a4 = byte_4FB4720;
        }
      }
    }
    *(_BYTE *)(v17 + 166) = a4;
    v49 = sub_16D5D50();
    v50 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v51 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v52 = v50[2];
          v53 = v50[3];
          if ( v49 <= v50[4] )
            break;
          v50 = (_QWORD *)v50[3];
          if ( !v53 )
            goto LABEL_43;
        }
        v51 = v50;
        v50 = (_QWORD *)v50[2];
      }
      while ( v52 );
LABEL_43:
      if ( v51 != dword_4FA0208 && v49 >= *((_QWORD *)v51 + 4) )
      {
        v54 = *((_QWORD *)v51 + 7);
        if ( v54 )
        {
          v55 = v51 + 12;
          do
          {
            while ( 1 )
            {
              v56 = *(_QWORD *)(v54 + 16);
              v57 = *(_QWORD *)(v54 + 24);
              if ( *(_DWORD *)(v54 + 32) >= dword_4FB43E8 )
                break;
              v54 = *(_QWORD *)(v54 + 24);
              if ( !v57 )
                goto LABEL_50;
            }
            v55 = (_DWORD *)v54;
            v54 = *(_QWORD *)(v54 + 16);
          }
          while ( v56 );
LABEL_50:
          if ( v51 + 12 != v55 && dword_4FB43E8 >= v55[8] && v55[9] )
            a5 = byte_4FB4480;
        }
      }
    }
    *(_BYTE *)(v17 + 167) = a5;
    *(_BYTE *)(v17 + 168) = a6;
    *(_BYTE *)(v17 + 169) = a7;
  }
  if ( v78 )
    v78(&v77, &v77, 3);
  return v17;
}
