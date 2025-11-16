// Function: sub_344D9B0
// Address: 0x344d9b0
//
__int64 __fastcall sub_344D9B0(__int128 a1, __m128i *a2, __int64 a3, __int32 a4)
{
  void (__fastcall *v5)(__m128i *, __m128i *, __int64); // rax
  __m128i *v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 result; // rax
  void (__fastcall *v11)(_BYTE *, __m128i *, __int64); // r8
  __int64 v12; // rax
  void (__fastcall *v13)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int8 (__fastcall *v16)(__m128i *, __m128i **); // rsi
  void (__fastcall *v17)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm2
  unsigned __int8 (__fastcall *v20)(__m128i *, __m128i **); // rsi
  __m128i v21; // xmm0
  __m128i v22; // xmm3
  __int64 v23; // rsi
  __int64 v24; // rbx
  int v25; // r8d
  __int64 v26; // rbx
  __int64 v27; // r8
  __int64 v28; // rbx
  void (__fastcall *v29)(__m128i *, __m128i *, __int64); // rax
  __m128i *v30; // r14
  __int64 v31; // r13
  __int64 v32; // r8
  __int64 v33; // r8
  int v34; // r8d
  bool v35; // zf
  __int64 v36; // r8
  int v37; // r8d
  __int64 v38; // [rsp+10h] [rbp-260h]
  const __m128i *v41; // [rsp+28h] [rbp-248h]
  __int32 v42; // [rsp+28h] [rbp-248h]
  __int128 v43; // [rsp+30h] [rbp-240h] BYREF
  __m128i *v44; // [rsp+40h] [rbp-230h] BYREF
  int v45; // [rsp+48h] [rbp-228h]
  __m128i *v46; // [rsp+50h] [rbp-220h] BYREF
  int v47; // [rsp+58h] [rbp-218h]
  __m128i *v48; // [rsp+60h] [rbp-210h] BYREF
  int v49; // [rsp+68h] [rbp-208h]
  __m128i *v50; // [rsp+70h] [rbp-200h] BYREF
  int v51; // [rsp+78h] [rbp-1F8h]
  __m128i *v52; // [rsp+80h] [rbp-1F0h] BYREF
  int v53; // [rsp+88h] [rbp-1E8h]
  __m128i *v54; // [rsp+90h] [rbp-1E0h] BYREF
  int v55; // [rsp+98h] [rbp-1D8h]
  __m128i *v56; // [rsp+A0h] [rbp-1D0h] BYREF
  int v57; // [rsp+A8h] [rbp-1C8h]
  __m128i v58; // [rsp+B0h] [rbp-1C0h] BYREF
  _BYTE v59[16]; // [rsp+C0h] [rbp-1B0h] BYREF
  void (__fastcall *v60)(_QWORD, _QWORD, _QWORD); // [rsp+D0h] [rbp-1A0h]
  __int64 v61; // [rsp+D8h] [rbp-198h]
  const __m128i *v62; // [rsp+E0h] [rbp-190h]
  _BYTE v63[16]; // [rsp+F0h] [rbp-180h] BYREF
  void (__fastcall *v64)(_QWORD, _QWORD, _QWORD); // [rsp+100h] [rbp-170h]
  unsigned __int8 (__fastcall *v65)(_QWORD, _QWORD); // [rsp+108h] [rbp-168h]
  const __m128i *v66; // [rsp+110h] [rbp-160h]
  _BYTE v67[16]; // [rsp+120h] [rbp-150h] BYREF
  void (__fastcall *v68)(_QWORD, _QWORD, _QWORD); // [rsp+130h] [rbp-140h]
  unsigned __int8 (__fastcall *v69)(_QWORD, _QWORD); // [rsp+138h] [rbp-138h]
  __int64 v70; // [rsp+140h] [rbp-130h]
  __m128i v71; // [rsp+150h] [rbp-120h] BYREF
  void (__fastcall *v72)(__m128i *, __m128i *, __int64); // [rsp+160h] [rbp-110h]
  unsigned __int8 (__fastcall *v73)(_QWORD, _QWORD); // [rsp+168h] [rbp-108h]
  __int64 v74; // [rsp+170h] [rbp-100h]
  __m128i v75; // [rsp+180h] [rbp-F0h] BYREF
  void (__fastcall *v76)(_QWORD, _QWORD, _QWORD); // [rsp+190h] [rbp-E0h]
  unsigned __int8 (__fastcall *v77)(__m128i *, __m128i **); // [rsp+198h] [rbp-D8h]
  __int64 v78; // [rsp+1A0h] [rbp-D0h]
  __m128i v79; // [rsp+1B0h] [rbp-C0h] BYREF
  void (__fastcall *v80)(__m128i *, __m128i *, __int64); // [rsp+1C0h] [rbp-B0h]
  unsigned __int8 (__fastcall *v81)(__m128i *, __m128i **); // [rsp+1C8h] [rbp-A8h]
  __int64 v82; // [rsp+1D0h] [rbp-A0h]
  __m128i v83; // [rsp+1E0h] [rbp-90h] BYREF
  void (__fastcall *v84)(_QWORD, _QWORD, _QWORD); // [rsp+1F0h] [rbp-80h]
  unsigned __int8 (__fastcall *v85)(__m128i *, __m128i **); // [rsp+1F8h] [rbp-78h]
  __int64 v86; // [rsp+200h] [rbp-70h]
  __m128i v87; // [rsp+210h] [rbp-60h] BYREF
  void (__fastcall *v88)(__m128i *, __m128i *, __int64); // [rsp+220h] [rbp-50h]
  unsigned __int8 (__fastcall *v89)(__m128i *, __m128i *); // [rsp+228h] [rbp-48h]

  v5 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a2[1].m128i_i64[0];
  v43 = a1;
  v88 = 0;
  if ( v5 )
  {
    v5(&v87, a2, 2);
    v89 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))a2[1].m128i_i64[1];
    v88 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a2[1].m128i_i64[0];
  }
  v6 = (__m128i *)&v43;
  v7 = (__int64)&v87;
  v41 = sub_344D5D0((__m128i *)&v43, &v87);
  if ( v88 )
  {
    v7 = (__int64)&v87;
    v6 = &v87;
    v88(&v87, &v87, 3);
  }
  v8 = v43;
  v9 = 16LL * *((_QWORD *)&v43 + 1);
  result = 16LL * *((_QWORD *)&v43 + 1) + v43;
  if ( v41 != (const __m128i *)result )
  {
    v60 = 0;
    v11 = (void (__fastcall *)(_BYTE *, __m128i *, __int64))a2[1].m128i_i64[0];
    if ( v11 )
    {
      v6 = (__m128i *)v59;
      v11(v59, a2, 2);
      v12 = a2[1].m128i_i64[1];
      v8 = (__int64)v41;
      v64 = 0;
      v61 = v12;
      v13 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a2[1].m128i_i64[0];
      v62 = v41;
      v60 = v13;
      if ( v13 )
      {
        v6 = (__m128i *)v63;
        v13(v63, v59, 2);
        v68 = 0;
        v14 = v43;
        v15 = (__int64)v62;
        v65 = (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))v61;
        v9 = 16LL * *((_QWORD *)&v43 + 1);
        v8 = v43 + 16LL * *((_QWORD *)&v43 + 1);
        v66 = v62;
        v64 = v60;
        v38 = v8;
        if ( v60 )
        {
          v6 = (__m128i *)v67;
          v60(v67, v63, 2);
          v15 = (__int64)v66;
          v72 = 0;
          v69 = v65;
          v70 = (__int64)v66;
          v68 = v64;
          if ( v64 )
          {
            v6 = &v71;
            v64(&v71, v67, 2);
            v16 = v69;
            v17 = v68;
            v15 = v70;
            goto LABEL_11;
          }
LABEL_74:
          v16 = v73;
          v17 = 0;
LABEL_11:
          v18 = _mm_loadu_si128(&v71);
          v19 = _mm_loadu_si128(&v87);
          v77 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i **))v16;
          v74 = v15;
          v20 = v81;
          v72 = 0;
          v73 = 0;
          v76 = v17;
          v78 = v15;
          v80 = 0;
          v71 = v19;
          v87 = v18;
          v75 = v18;
          if ( v17 )
          {
            v6 = &v79;
            v17(&v79, &v75, 2);
            v20 = v77;
            v17 = v76;
            v15 = v78;
          }
          v21 = _mm_loadu_si128(&v79);
          v22 = _mm_loadu_si128(&v87);
          v85 = v20;
          v23 = v9;
          v24 = v9 >> 6;
          v84 = v17;
          v7 = v23 >> 4;
          v82 = v15;
          v80 = 0;
          v81 = 0;
          v86 = v15;
          v79 = v22;
          v87 = v21;
          v83 = v21;
          if ( v24 > 0 )
          {
            v6 = *(__m128i **)v14;
            v7 = *(_QWORD *)v15;
            v25 = *(_DWORD *)(v14 + 8);
            v26 = v14 + (v24 << 6);
            if ( *(_QWORD *)v15 == *(_QWORD *)v14 )
              goto LABEL_59;
LABEL_15:
            v44 = v6;
            v45 = v25;
            if ( !v17 )
              goto LABEL_71;
            v6 = &v83;
            v7 = (__int64)&v44;
            if ( !v85(&v83, &v44) )
            {
LABEL_21:
              v17 = v84;
              goto LABEL_22;
            }
            v15 = v86;
            v6 = *(__m128i **)(v14 + 16);
            v17 = v84;
            v27 = *(unsigned int *)(v14 + 24);
            v7 = *(_QWORD *)v86;
            if ( v6 != *(__m128i **)v86 )
            {
LABEL_18:
              v46 = v6;
              v47 = v27;
              if ( !v17 )
                goto LABEL_71;
              v7 = (__int64)&v46;
              v6 = &v83;
              if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i **, __int64, __int64, __int64))v85)(
                      &v83,
                      &v46,
                      v15,
                      v8,
                      v27) )
              {
                v14 += 16;
                goto LABEL_21;
              }
              v15 = v86;
              v6 = *(__m128i **)(v14 + 32);
              v17 = v84;
              v32 = *(unsigned int *)(v14 + 40);
              v7 = *(_QWORD *)v86;
              if ( v6 == *(__m128i **)v86 )
                goto LABEL_63;
LABEL_49:
              v48 = v6;
              v49 = v32;
              if ( v17 )
              {
                v7 = (__int64)&v48;
                v6 = &v83;
                if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i **, __int64, __int64, __int64))v85)(
                        &v83,
                        &v48,
                        v15,
                        v8,
                        v32) )
                {
                  v17 = v84;
                  v14 += 32;
                  goto LABEL_22;
                }
                v15 = v86;
                v6 = *(__m128i **)(v14 + 48);
                v17 = v84;
                v33 = *(unsigned int *)(v14 + 56);
                v7 = *(_QWORD *)v86;
                if ( v6 == *(__m128i **)v86 )
                  goto LABEL_65;
LABEL_53:
                v50 = v6;
                v51 = v33;
                if ( v17 )
                {
                  v7 = (__int64)&v50;
                  v6 = &v83;
                  if ( ((unsigned __int8 (__fastcall *)(__m128i *, __m128i **, __int64, __int64, __int64))v85)(
                         &v83,
                         &v50,
                         v15,
                         v8,
                         v33) )
                  {
                    v17 = v84;
                    goto LABEL_57;
                  }
                  v17 = v84;
                  v14 += 48;
                  goto LABEL_22;
                }
              }
LABEL_71:
              sub_4263D6(v6, v7, v15);
            }
            while ( 1 )
            {
              if ( (_DWORD)v27 != *(_DWORD *)(v15 + 8) )
                goto LABEL_18;
              v6 = *(__m128i **)(v14 + 32);
              v32 = *(unsigned int *)(v14 + 40);
              if ( v6 != (__m128i *)v7 )
                goto LABEL_49;
LABEL_63:
              if ( *(_DWORD *)(v15 + 8) != (_DWORD)v32 )
                goto LABEL_49;
              v6 = *(__m128i **)(v14 + 48);
              v33 = *(unsigned int *)(v14 + 56);
              if ( v6 != (__m128i *)v7 )
                goto LABEL_53;
LABEL_65:
              if ( *(_DWORD *)(v15 + 8) != (_DWORD)v33 )
                goto LABEL_53;
LABEL_57:
              v14 += 64;
              if ( v26 == v14 )
                break;
              v15 = v86;
              v6 = *(__m128i **)v14;
              v25 = *(_DWORD *)(v14 + 8);
              v7 = *(_QWORD *)v86;
              if ( *(_QWORD *)v86 != *(_QWORD *)v14 )
                goto LABEL_15;
LABEL_59:
              if ( v25 != *(_DWORD *)(v15 + 8) )
                goto LABEL_15;
              v6 = *(__m128i **)(v14 + 16);
              v27 = *(unsigned int *)(v14 + 24);
              if ( v6 != (__m128i *)v7 )
                goto LABEL_18;
            }
            v7 = (v38 - v14) >> 4;
          }
          switch ( v7 )
          {
            case 2LL:
              v15 = v86;
              v6 = *(__m128i **)v86;
              break;
            case 3LL:
              v15 = v86;
              v7 = *(_QWORD *)v14;
              v36 = *(unsigned int *)(v14 + 8);
              v6 = *(__m128i **)v86;
              if ( *(_QWORD *)v14 != *(_QWORD *)v86 || (_DWORD)v36 != *(_DWORD *)(v86 + 8) )
              {
                v52 = *(__m128i **)v14;
                v53 = v36;
                if ( !v17 )
                  goto LABEL_71;
                v7 = (__int64)&v52;
                v6 = &v83;
                if ( !((unsigned __int8 (__fastcall *)(__m128i *, __m128i **, __int64, __int64, __int64))v85)(
                        &v83,
                        &v52,
                        v86,
                        v8,
                        v36) )
                  goto LABEL_21;
                v15 = v86;
                v17 = v84;
                v6 = *(__m128i **)v86;
              }
              v14 += 16;
              break;
            case 1LL:
              v15 = v86;
              v7 = *(_QWORD *)v86;
LABEL_82:
              v6 = *(__m128i **)v14;
              v34 = *(_DWORD *)(v14 + 8);
              if ( *(_QWORD *)v14 != v7 || *(_DWORD *)(v15 + 8) != v34 )
              {
                v56 = *(__m128i **)v14;
                v57 = v34;
                if ( !v17 )
                  goto LABEL_71;
                v7 = (__int64)&v56;
                v6 = &v83;
                v35 = v85(&v83, &v56) == 0;
                v17 = v84;
                if ( v35 )
                {
LABEL_22:
                  if ( v17 )
                  {
                    v6 = &v83;
                    v7 = (__int64)&v83;
                    v17(&v83, &v83, 3);
                  }
                  if ( v80 )
                  {
                    v6 = &v79;
                    v7 = (__int64)&v79;
                    v80(&v79, &v79, 3);
                  }
                  if ( v76 )
                  {
                    v6 = &v75;
                    v7 = (__int64)&v75;
                    v76(&v75, &v75, 3);
                  }
                  if ( v72 )
                  {
                    v6 = &v71;
                    v7 = (__int64)&v71;
                    v72(&v71, &v71, 3);
                  }
                  if ( v68 )
                  {
                    v6 = (__m128i *)v67;
                    v7 = (__int64)v67;
                    v68(v67, v67, 3);
                  }
                  if ( v64 )
                  {
                    v6 = (__m128i *)v63;
                    v7 = (__int64)v63;
                    v64(v63, v63, 3);
                  }
                  result = (__int64)v60;
                  if ( v60 )
                  {
                    v6 = (__m128i *)v59;
                    v7 = (__int64)v59;
                    result = ((__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))v60)(v59, v59, 3);
                  }
                  if ( v38 == v14 )
                  {
                    v28 = v41->m128i_i64[0];
                    result = v41->m128i_u32[2];
                    v42 = v41->m128i_i32[2];
                    if ( v28 )
                      goto LABEL_39;
                  }
                  goto LABEL_37;
                }
              }
LABEL_79:
              v14 = v38;
              goto LABEL_22;
            default:
              goto LABEL_79;
          }
          v7 = *(_QWORD *)v14;
          v37 = *(_DWORD *)(v14 + 8);
          if ( v6 != *(__m128i **)v14 || v37 != *(_DWORD *)(v15 + 8) )
          {
            v54 = *(__m128i **)v14;
            v55 = v37;
            if ( !v17 )
              goto LABEL_71;
            v7 = (__int64)&v54;
            v6 = &v83;
            if ( !v85(&v83, &v54) )
              goto LABEL_21;
            v15 = v86;
            v17 = v84;
            v7 = *(_QWORD *)v86;
          }
          v14 += 16;
          goto LABEL_82;
        }
      }
      else
      {
        v15 = (__int64)v41;
        v14 = v43;
        v66 = v41;
        v68 = 0;
        v9 = 16LL * *((_QWORD *)&v43 + 1);
        v38 = v43 + 16LL * *((_QWORD *)&v43 + 1);
      }
    }
    else
    {
      v15 = (__int64)v41;
      v14 = v43;
      v64 = 0;
      v38 = 16LL * *((_QWORD *)&v43 + 1) + v43;
      v62 = v41;
      v66 = v41;
      v68 = 0;
    }
    v70 = v15;
    goto LABEL_74;
  }
LABEL_37:
  if ( !a3 )
    return result;
  v28 = a3;
  v42 = a4;
LABEL_39:
  v29 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a2[1].m128i_i64[0];
  v88 = 0;
  if ( v29 )
  {
    v7 = (__int64)a2;
    v6 = &v87;
    v29(&v87, a2, 2);
    v30 = (__m128i *)v43;
    v89 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))a2[1].m128i_i64[1];
    result = a2[1].m128i_i64[0];
    v88 = (void (__fastcall *)(__m128i *, __m128i *, __int64))result;
    v31 = v43 + 16LL * *((_QWORD *)&v43 + 1);
    if ( (_QWORD)v43 != v31 )
    {
      do
      {
        v58 = _mm_loadu_si128(v30);
        if ( !result )
          goto LABEL_71;
        v7 = (__int64)&v58;
        v6 = &v87;
        if ( v89(&v87, &v58) )
        {
          v30->m128i_i64[0] = v28;
          v30->m128i_i32[2] = v42;
        }
        ++v30;
        result = (__int64)v88;
      }
      while ( (__m128i *)v31 != v30 );
    }
    if ( result )
      return ((__int64 (__fastcall *)(__m128i *, __m128i *, __int64))result)(&v87, &v87, 3);
  }
  else
  {
    v15 = v43;
    result = 16LL * *((_QWORD *)&v43 + 1);
    if ( 16LL * *((_QWORD *)&v43 + 1) )
    {
      v58 = _mm_loadu_si128((const __m128i *)v43);
      goto LABEL_71;
    }
  }
  return result;
}
