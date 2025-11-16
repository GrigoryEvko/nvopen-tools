// Function: sub_2A807C0
// Address: 0x2a807c0
//
__int64 __fastcall sub_2A807C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rax
  unsigned __int8 *v5; // r12
  __int64 v6; // rsi
  __int64 v7; // rdx
  char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  const char *v11; // rax
  size_t v12; // rdx
  char *v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rdi
  __m128i v17; // rax
  char v18; // al
  _QWORD *v19; // rdx
  char v20; // al
  __m128i *v21; // rcx
  char v22; // dl
  _QWORD *v23; // rsi
  char v24; // al
  const char *v25; // rsi
  _QWORD *v26; // rcx
  char v27; // dl
  const char *v28; // rsi
  _QWORD *v29; // rcx
  __m128i v30; // xmm0
  __m128i v31; // xmm1
  _DWORD *v32; // rdi
  __m128i *v33; // rsi
  __int64 i; // rcx
  _DWORD *v35; // rdi
  _DWORD *v36; // rsi
  __int64 j; // rcx
  const char **v38; // rsi
  __int64 v39; // rcx
  const char **v40; // rdi
  _DWORD *v41; // rsi
  __int64 v42; // rcx
  const char **v43; // rdi
  _DWORD *v44; // rdi
  __int64 **v45; // rsi
  __int64 k; // rcx
  __int64 v47; // [rsp+0h] [rbp-290h]
  const char *v48; // [rsp+8h] [rbp-288h]
  __int64 v49; // [rsp+10h] [rbp-280h]
  const char *v50; // [rsp+18h] [rbp-278h]
  __int64 v51; // [rsp+20h] [rbp-270h]
  __int64 v52; // [rsp+28h] [rbp-268h]
  __int64 v53; // [rsp+30h] [rbp-260h]
  unsigned __int8 v54; // [rsp+3Fh] [rbp-251h]
  __int64 v55; // [rsp+48h] [rbp-248h]
  void *s2; // [rsp+58h] [rbp-238h]
  _QWORD *v58; // [rsp+70h] [rbp-220h] BYREF
  __int64 v59; // [rsp+78h] [rbp-218h]
  _BYTE v60[16]; // [rsp+80h] [rbp-210h] BYREF
  __m128i v61; // [rsp+90h] [rbp-200h] BYREF
  _QWORD v62[2]; // [rsp+A0h] [rbp-1F0h] BYREF
  __m128i v63; // [rsp+B0h] [rbp-1E0h] BYREF
  __m128i v64; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v65; // [rsp+D0h] [rbp-1C0h]
  _QWORD v66[4]; // [rsp+E0h] [rbp-1B0h] BYREF
  char v67; // [rsp+100h] [rbp-190h]
  char v68; // [rsp+101h] [rbp-18Fh]
  __m128i v69; // [rsp+110h] [rbp-180h] BYREF
  __m128i v70; // [rsp+120h] [rbp-170h]
  __int64 v71; // [rsp+130h] [rbp-160h]
  _QWORD v72[4]; // [rsp+140h] [rbp-150h] BYREF
  __int16 v73; // [rsp+160h] [rbp-130h]
  __int64 *v74; // [rsp+170h] [rbp-120h] BYREF
  const char *v75; // [rsp+178h] [rbp-118h]
  _QWORD *v76; // [rsp+180h] [rbp-110h]
  __int64 v77; // [rsp+188h] [rbp-108h]
  __int16 v78; // [rsp+190h] [rbp-100h]
  _QWORD v79[4]; // [rsp+1A0h] [rbp-F0h] BYREF
  char v80; // [rsp+1C0h] [rbp-D0h]
  char v81; // [rsp+1C1h] [rbp-CFh]
  const char *v82; // [rsp+1D0h] [rbp-C0h] BYREF
  const char *v83; // [rsp+1D8h] [rbp-B8h]
  _QWORD *v84; // [rsp+1E0h] [rbp-B0h]
  __int64 v85; // [rsp+1E8h] [rbp-A8h]
  __int16 v86; // [rsp+1F0h] [rbp-A0h]
  _QWORD v87[4]; // [rsp+200h] [rbp-90h] BYREF
  __int16 v88; // [rsp+220h] [rbp-70h]
  const char *v89[2]; // [rsp+230h] [rbp-60h] BYREF
  _QWORD v90[2]; // [rsp+240h] [rbp-50h] BYREF
  __int16 v91; // [rsp+250h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 32);
  v55 = a2 + 24;
  if ( v2 != a2 + 24 )
  {
    v54 = 0;
    while ( 1 )
    {
      v5 = (unsigned __int8 *)(v2 - 56);
      v6 = *(_QWORD *)(a1 + 16);
      if ( !v2 )
        v5 = 0;
      v7 = *(_QWORD *)(a1 + 24);
      v60[0] = 0;
      v58 = v60;
      v59 = 0;
      sub_C88F40((__int64)v89, v6, v7, 0);
      v8 = (char *)sub_BD5D20((__int64)v5);
      sub_C894D0(&v61, v89, *(char **)(a1 + 48), *(_QWORD *)(a1 + 56), v8, v9, &v58);
      sub_C88FF0(v89);
      if ( v59 )
      {
        v88 = 260;
        v87[0] = &v58;
        v79[0] = ": ";
        v81 = 1;
        v80 = 3;
        v72[0] = a2 + 168;
        v73 = 260;
        v68 = 1;
        v66[0] = " in ";
        v67 = 3;
        v17.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v5);
        v63.m128i_i64[0] = (__int64)"unable to transforn ";
        v64 = v17;
        v18 = v67;
        LOWORD(v65) = 1283;
        if ( v67 )
        {
          if ( v67 == 1 )
          {
            v30 = _mm_loadu_si128(&v63);
            v31 = _mm_loadu_si128(&v64);
            v71 = v65;
            v20 = v73;
            v69 = v30;
            v70 = v31;
            if ( (_BYTE)v73 )
            {
              if ( (_BYTE)v73 == 1 )
                goto LABEL_60;
              if ( BYTE1(v71) == 1 )
              {
                v21 = (__m128i *)v69.m128i_i64[0];
                v22 = 3;
                v52 = v69.m128i_i64[1];
              }
              else
              {
LABEL_33:
                v21 = &v69;
                v22 = 2;
              }
              if ( HIBYTE(v73) == 1 )
              {
                v23 = (_QWORD *)v72[0];
                v51 = v72[1];
              }
              else
              {
                v23 = v72;
                v20 = 2;
              }
              HIBYTE(v78) = v20;
              v24 = v80;
              v74 = (__int64 *)v21;
              v75 = (const char *)v52;
              v76 = v23;
              v77 = v51;
              LOBYTE(v78) = v22;
              if ( v80 )
                goto LABEL_37;
              goto LABEL_54;
            }
          }
          else
          {
            if ( v68 == 1 )
            {
              v19 = (_QWORD *)v66[0];
              v53 = v66[1];
            }
            else
            {
              v19 = v66;
              v18 = 2;
            }
            BYTE1(v71) = v18;
            v20 = v73;
            v69.m128i_i64[0] = (__int64)&v63;
            v70.m128i_i64[0] = (__int64)v19;
            v70.m128i_i64[1] = v53;
            LOBYTE(v71) = 2;
            if ( (_BYTE)v73 )
            {
              if ( (_BYTE)v73 != 1 )
                goto LABEL_33;
LABEL_60:
              v32 = &v74;
              v33 = &v69;
              for ( i = 10; i; --i )
              {
                *v32 = v33->m128i_i32[0];
                v33 = (__m128i *)((char *)v33 + 4);
                ++v32;
              }
              v22 = v78;
              if ( (_BYTE)v78 )
              {
                v24 = v80;
                if ( v80 )
                {
                  if ( (_BYTE)v78 == 1 )
                  {
                    v35 = &v82;
                    v36 = v79;
                    for ( j = 10; j; --j )
                      *v35++ = *v36++;
                    goto LABEL_43;
                  }
LABEL_37:
                  if ( v24 == 1 )
                  {
                    v44 = &v82;
                    v45 = &v74;
                    for ( k = 10; k; --k )
                    {
                      *v44 = *(_DWORD *)v45;
                      v45 = (__int64 **)((char *)v45 + 4);
                      ++v44;
                    }
                    v24 = v86;
                    if ( (_BYTE)v86 )
                      goto LABEL_43;
                  }
                  else
                  {
                    if ( HIBYTE(v78) == 1 )
                    {
                      v25 = (const char *)v74;
                      v50 = v75;
                    }
                    else
                    {
                      v25 = (const char *)&v74;
                      v22 = 2;
                    }
                    if ( v81 == 1 )
                    {
                      v26 = (_QWORD *)v79[0];
                      v49 = v79[1];
                    }
                    else
                    {
                      v26 = v79;
                      v24 = 2;
                    }
                    v82 = v25;
                    v84 = v26;
                    v83 = v50;
                    LOBYTE(v86) = v22;
                    v85 = v49;
                    HIBYTE(v86) = v24;
                    v24 = v22;
LABEL_43:
                    v27 = v88;
                    if ( (_BYTE)v88 )
                    {
                      if ( v24 == 1 )
                      {
                        v41 = v87;
                        v42 = 10;
                        v43 = v89;
                        while ( v42 )
                        {
                          *(_DWORD *)v43 = *v41++;
                          v43 = (const char **)((char *)v43 + 4);
                          --v42;
                        }
                      }
                      else if ( (_BYTE)v88 == 1 )
                      {
                        v38 = &v82;
                        v39 = 10;
                        v40 = v89;
                        while ( v39 )
                        {
                          *(_DWORD *)v40 = *(_DWORD *)v38;
                          v38 = (const char **)((char *)v38 + 4);
                          v40 = (const char **)((char *)v40 + 4);
                          --v39;
                        }
                      }
                      else
                      {
                        if ( HIBYTE(v86) == 1 )
                        {
                          v28 = v82;
                          v48 = v83;
                        }
                        else
                        {
                          v28 = (const char *)&v82;
                          v24 = 2;
                        }
                        if ( HIBYTE(v88) == 1 )
                        {
                          v29 = (_QWORD *)v87[0];
                          v47 = v87[1];
                        }
                        else
                        {
                          v29 = v87;
                          v27 = 2;
                        }
                        v89[0] = v28;
                        v90[0] = v29;
                        v89[1] = v48;
                        LOBYTE(v91) = v24;
                        v90[1] = v47;
                        HIBYTE(v91) = v27;
                      }
LABEL_51:
                      sub_C64D30((__int64)v89, 1u);
                    }
                  }
LABEL_55:
                  v91 = 256;
                  goto LABEL_51;
                }
              }
LABEL_54:
              v86 = 256;
              goto LABEL_55;
            }
          }
        }
        else
        {
          LOWORD(v71) = 256;
        }
        v78 = 256;
        goto LABEL_54;
      }
      v10 = v61.m128i_i64[1];
      s2 = (void *)v61.m128i_i64[0];
      v11 = sub_BD5D20((__int64)v5);
      if ( v12 == v10 && (!v12 || !memcmp(v11, s2, v12)) )
      {
        if ( (_QWORD *)v61.m128i_i64[0] != v62 )
          j_j___libc_free_0(v61.m128i_u64[0]);
        if ( v58 == (_QWORD *)v60 )
          goto LABEL_9;
        j_j___libc_free_0((unsigned __int64)v58);
        v2 = *(_QWORD *)(v2 + 8);
        if ( v55 == v2 )
          return v54;
      }
      else
      {
        if ( v5 )
        {
          v13 = (char *)sub_BD5D20((__int64)v5);
          v89[0] = (const char *)v90;
          sub_2A7FC90((__int64 *)v89, v13, (__int64)&v13[v14]);
          sub_2A7FD40(a2, (__int64)v5, (__int64)v89, (__int64)&v61);
          if ( (_QWORD *)v89[0] != v90 )
            j_j___libc_free_0((unsigned __int64)v89[0]);
        }
        v15 = sub_BA8CB0(a2, v61.m128i_i64[0], v61.m128i_u64[1]);
        if ( v15 )
        {
          v4 = sub_BD5C70((__int64)v15);
          sub_BD6500((__int64)v5, v4);
        }
        else
        {
          v91 = 260;
          v89[0] = (const char *)&v61;
          sub_BD6B50(v5, v89);
        }
        if ( (_QWORD *)v61.m128i_i64[0] != v62 )
          j_j___libc_free_0(v61.m128i_u64[0]);
        if ( v58 != (_QWORD *)v60 )
          j_j___libc_free_0((unsigned __int64)v58);
        v54 = 1;
LABEL_9:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v55 == v2 )
          return v54;
      }
    }
  }
  return 0;
}
