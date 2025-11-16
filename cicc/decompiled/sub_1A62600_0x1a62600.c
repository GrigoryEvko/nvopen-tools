// Function: sub_1A62600
// Address: 0x1a62600
//
__int64 sub_1A62600()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _DWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // edx
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _DWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  char v18; // dl
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  char v24; // dl
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rax
  _DWORD *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  char v30; // dl
  unsigned __int64 v31; // rsi
  _QWORD *v32; // rax
  _DWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  char v36; // dl
  __int64 v37; // rax
  _DWORD *v38; // r8
  _DWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v43; // rax
  _DWORD *v44; // r8
  _DWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  _DWORD *v49; // r8
  _DWORD *v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  _DWORD *v54; // r8
  _DWORD *v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rax
  _DWORD *v59; // r8
  _DWORD *v60; // rdi
  __int64 v61; // rcx
  __int64 v62; // rdx
  __m128i v63; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v64)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v65; // [rsp+18h] [rbp-18h]

  v64 = 0;
  v0 = sub_22077B0(216);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = _mm_loadu_si128(&v63);
    *(_QWORD *)(v0 + 16) = &unk_4FB43CC;
    v3 = _mm_loadu_si128((const __m128i *)(v0 + 184));
    *(_QWORD *)(v0 + 80) = v0 + 64;
    v4 = *(_QWORD *)(v0 + 208);
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F5528;
    *(_QWORD *)(v0 + 160) = 0x1000000000001LL;
    v63 = v3;
    *(_QWORD *)(v0 + 200) = v64;
    v5 = v65;
    *(__m128i *)(v1 + 184) = v2;
    v65 = v4;
    *(_DWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    v64 = 0;
    *(_QWORD *)(v1 + 208) = v5;
    v6 = sub_163A1D0();
    sub_1A62500(v6);
    v7 = sub_16D5D50();
    v8 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v9 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v10 = v8[2];
          v11 = v8[3];
          if ( v7 <= v8[4] )
            break;
          v8 = (_QWORD *)v8[3];
          if ( !v11 )
            goto LABEL_7;
        }
        v9 = v8;
        v8 = (_QWORD *)v8[2];
      }
      while ( v10 );
LABEL_7:
      v12 = 1;
      if ( v9 != dword_4FA0208 && v7 >= *((_QWORD *)v9 + 4) )
      {
        v43 = *((_QWORD *)v9 + 7);
        v44 = v9 + 12;
        if ( v43 )
        {
          v45 = v9 + 12;
          do
          {
            while ( 1 )
            {
              v46 = *(_QWORD *)(v43 + 16);
              v47 = *(_QWORD *)(v43 + 24);
              if ( *(_DWORD *)(v43 + 32) >= dword_4FB4768 )
                break;
              v43 = *(_QWORD *)(v43 + 24);
              if ( !v47 )
                goto LABEL_53;
            }
            v45 = (_DWORD *)v43;
            v43 = *(_QWORD *)(v43 + 16);
          }
          while ( v46 );
LABEL_53:
          v12 = 1;
          if ( v44 != v45 && dword_4FB4768 >= v45[8] )
          {
            v12 = 1;
            if ( v45[9] )
              v12 = dword_4FB4800;
          }
        }
      }
    }
    else
    {
      v12 = 1;
    }
    *(_DWORD *)(v1 + 160) = v12;
    v13 = sub_16D5D50();
    v14 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v15 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v16 = v14[2];
          v17 = v14[3];
          if ( v13 <= v14[4] )
            break;
          v14 = (_QWORD *)v14[3];
          if ( !v17 )
            goto LABEL_14;
        }
        v15 = v14;
        v14 = (_QWORD *)v14[2];
      }
      while ( v16 );
LABEL_14:
      v18 = 0;
      if ( v15 != dword_4FA0208 && v13 >= *((_QWORD *)v15 + 4) )
      {
        v48 = *((_QWORD *)v15 + 7);
        v49 = v15 + 12;
        if ( v48 )
        {
          v50 = v15 + 12;
          do
          {
            while ( 1 )
            {
              v51 = *(_QWORD *)(v48 + 16);
              v52 = *(_QWORD *)(v48 + 24);
              if ( *(_DWORD *)(v48 + 32) >= dword_4FB44C8 )
                break;
              v48 = *(_QWORD *)(v48 + 24);
              if ( !v52 )
                goto LABEL_63;
            }
            v50 = (_DWORD *)v48;
            v48 = *(_QWORD *)(v48 + 16);
          }
          while ( v51 );
LABEL_63:
          v18 = 0;
          if ( v49 != v50 && dword_4FB44C8 >= v50[8] )
          {
            v18 = byte_4FB4560;
            if ( !v50[9] )
              v18 = 0;
          }
        }
      }
    }
    else
    {
      v18 = 0;
    }
    *(_BYTE *)(v1 + 164) = v18;
    v19 = sub_16D5D50();
    v20 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v21 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v22 = v20[2];
          v23 = v20[3];
          if ( v19 <= v20[4] )
            break;
          v20 = (_QWORD *)v20[3];
          if ( !v23 )
            goto LABEL_21;
        }
        v21 = v20;
        v20 = (_QWORD *)v20[2];
      }
      while ( v22 );
LABEL_21:
      v24 = 0;
      if ( v21 != dword_4FA0208 && v19 >= *((_QWORD *)v21 + 4) )
      {
        v53 = *((_QWORD *)v21 + 7);
        v54 = v21 + 12;
        if ( v53 )
        {
          v55 = v21 + 12;
          do
          {
            while ( 1 )
            {
              v56 = *(_QWORD *)(v53 + 16);
              v57 = *(_QWORD *)(v53 + 24);
              if ( *(_DWORD *)(v53 + 32) >= dword_4FB45A8 )
                break;
              v53 = *(_QWORD *)(v53 + 24);
              if ( !v57 )
                goto LABEL_73;
            }
            v55 = (_DWORD *)v53;
            v53 = *(_QWORD *)(v53 + 16);
          }
          while ( v56 );
LABEL_73:
          v24 = 0;
          if ( v54 != v55 && dword_4FB45A8 >= v55[8] )
          {
            v24 = byte_4FB4640;
            if ( !v55[9] )
              v24 = 0;
          }
        }
      }
    }
    else
    {
      v24 = 0;
    }
    *(_BYTE *)(v1 + 165) = v24;
    v25 = sub_16D5D50();
    v26 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v27 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v28 = v26[2];
          v29 = v26[3];
          if ( v25 <= v26[4] )
            break;
          v26 = (_QWORD *)v26[3];
          if ( !v29 )
            goto LABEL_28;
        }
        v27 = v26;
        v26 = (_QWORD *)v26[2];
      }
      while ( v28 );
LABEL_28:
      v30 = 1;
      if ( v27 != dword_4FA0208 && v25 >= *((_QWORD *)v27 + 4) )
      {
        v58 = *((_QWORD *)v27 + 7);
        v59 = v27 + 12;
        if ( v58 )
        {
          v60 = v27 + 12;
          do
          {
            while ( 1 )
            {
              v61 = *(_QWORD *)(v58 + 16);
              v62 = *(_QWORD *)(v58 + 24);
              if ( *(_DWORD *)(v58 + 32) >= dword_4FB4688 )
                break;
              v58 = *(_QWORD *)(v58 + 24);
              if ( !v62 )
                goto LABEL_83;
            }
            v60 = (_DWORD *)v58;
            v58 = *(_QWORD *)(v58 + 16);
          }
          while ( v61 );
LABEL_83:
          v30 = 1;
          if ( v59 != v60 && dword_4FB4688 >= v60[8] )
          {
            v30 = byte_4FB4720;
            if ( !v60[9] )
              v30 = 1;
          }
        }
      }
    }
    else
    {
      v30 = 1;
    }
    *(_BYTE *)(v1 + 166) = v30;
    v31 = sub_16D5D50();
    v32 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v33 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v34 = v32[2];
          v35 = v32[3];
          if ( v31 <= v32[4] )
            break;
          v32 = (_QWORD *)v32[3];
          if ( !v35 )
            goto LABEL_35;
        }
        v33 = v32;
        v32 = (_QWORD *)v32[2];
      }
      while ( v34 );
LABEL_35:
      v36 = 0;
      if ( v33 != dword_4FA0208 && v31 >= *((_QWORD *)v33 + 4) )
      {
        v37 = *((_QWORD *)v33 + 7);
        v38 = v33 + 12;
        if ( v37 )
        {
          v39 = v33 + 12;
          do
          {
            while ( 1 )
            {
              v40 = *(_QWORD *)(v37 + 16);
              v41 = *(_QWORD *)(v37 + 24);
              if ( *(_DWORD *)(v37 + 32) >= dword_4FB43E8 )
                break;
              v37 = *(_QWORD *)(v37 + 24);
              if ( !v41 )
                goto LABEL_42;
            }
            v39 = (_DWORD *)v37;
            v37 = *(_QWORD *)(v37 + 16);
          }
          while ( v40 );
LABEL_42:
          v36 = 0;
          if ( v38 != v39 && dword_4FB43E8 >= v39[8] )
          {
            v36 = byte_4FB4480;
            if ( !v39[9] )
              v36 = 0;
          }
        }
      }
    }
    else
    {
      v36 = 0;
    }
    *(_BYTE *)(v1 + 167) = v36;
    *(_WORD *)(v1 + 168) = 256;
  }
  if ( v64 )
    v64(&v63, &v63, 3);
  return v1;
}
