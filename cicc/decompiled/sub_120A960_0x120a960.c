// Function: sub_120A960
// Address: 0x120a960
//
__int64 __fastcall sub_120A960(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  __int64 v6; // r11
  __int64 v10; // rbx
  __int64 v11; // rsi
  char v12; // cl
  char *v13; // r10
  char v15; // cl
  __m128i *v16; // rdx
  char v17; // cl
  char v18; // dl
  char v19; // dl
  _QWORD *v20; // rcx
  char v21; // cl
  char v22; // dl
  __m128i v23; // xmm1
  __m128i *v24; // rdi
  _QWORD *v25; // rsi
  char v26; // cl
  __m128i *v27; // rdi
  __m128i *v28; // rsi
  __int64 v29; // rax
  __m128i v30; // xmm5
  __m128i v31; // xmm7
  __m128i v32; // xmm7
  __int64 v33; // [rsp+8h] [rbp-258h]
  __int64 v34; // [rsp+10h] [rbp-250h]
  __int64 v35; // [rsp+18h] [rbp-248h]
  __int64 v36; // [rsp+20h] [rbp-240h]
  __int64 v37; // [rsp+28h] [rbp-238h]
  __int64 v38; // [rsp+30h] [rbp-230h]
  __int64 v39; // [rsp+38h] [rbp-228h]
  __int64 v40[2]; // [rsp+40h] [rbp-220h] BYREF
  __int64 v41; // [rsp+50h] [rbp-210h] BYREF
  __int64 v42[2]; // [rsp+60h] [rbp-200h] BYREF
  __int64 v43; // [rsp+70h] [rbp-1F0h] BYREF
  _QWORD v44[4]; // [rsp+80h] [rbp-1E0h] BYREF
  __int16 v45; // [rsp+A0h] [rbp-1C0h]
  _QWORD v46[4]; // [rsp+B0h] [rbp-1B0h] BYREF
  __int16 v47; // [rsp+D0h] [rbp-190h]
  __m128i v48; // [rsp+E0h] [rbp-180h] BYREF
  __m128i v49; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v50; // [rsp+100h] [rbp-160h]
  _QWORD v51[4]; // [rsp+110h] [rbp-150h] BYREF
  char v52; // [rsp+130h] [rbp-130h]
  char v53; // [rsp+131h] [rbp-12Fh]
  __m128i v54; // [rsp+140h] [rbp-120h] BYREF
  __m128i v55; // [rsp+150h] [rbp-110h] BYREF
  __int64 v56; // [rsp+160h] [rbp-100h]
  _QWORD v57[2]; // [rsp+170h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+190h] [rbp-D0h]
  __m128i v59; // [rsp+1A0h] [rbp-C0h] BYREF
  __m128i v60; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+1C0h] [rbp-A0h]
  __m128i v62; // [rsp+1D0h] [rbp-90h] BYREF
  __m128i v63; // [rsp+1E0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+1F0h] [rbp-70h]
  __m128i v65; // [rsp+200h] [rbp-60h] BYREF
  __m128i v66; // [rsp+210h] [rbp-50h]
  __int64 v67; // [rsp+220h] [rbp-40h]

  if ( a4 == *(_QWORD *)(a5 + 8) )
    return a5;
  v10 = a3;
  if ( *(_BYTE *)(a4 + 8) != 8 )
  {
    v62.m128i_i64[0] = (__int64)"'";
    LOWORD(v64) = 259;
    sub_1207630(v42, a4);
    v11 = *(_QWORD *)(a5 + 8);
    v57[0] = v42;
    v53 = 1;
    LOWORD(v58) = 260;
    v51[0] = "' but expected '";
    v52 = 3;
    sub_1207630(v40, v11);
    v12 = *(_BYTE *)(v10 + 32);
    v13 = "'";
    if ( v12 )
    {
      if ( v12 == 1 )
      {
        v18 = 3;
        v44[0] = "'";
        v45 = 259;
        v39 = v44[1];
      }
      else
      {
        if ( *(_BYTE *)(v10 + 33) == 1 )
        {
          v29 = *(_QWORD *)(v10 + 8);
          v10 = *(_QWORD *)v10;
          v38 = v29;
        }
        else
        {
          v12 = 2;
        }
        v44[0] = "'";
        v18 = 2;
        v13 = (char *)v44;
        v44[2] = v10;
        v44[3] = v38;
        LOBYTE(v45) = 3;
        HIBYTE(v45) = v12;
      }
      LOBYTE(v47) = v18;
      v19 = v52;
      v46[0] = v13;
      v46[1] = v39;
      v46[2] = "' defined with type '";
      HIBYTE(v47) = 3;
      v48 = (__m128i)(unsigned __int64)v46;
      v49.m128i_i64[0] = (__int64)v40;
      LOWORD(v50) = 1026;
      if ( v52 )
      {
        if ( v52 == 1 )
        {
          v23 = _mm_loadu_si128(&v49);
          v22 = v50;
          v54 = _mm_loadu_si128(&v48);
          v56 = v50;
          v55 = v23;
          v21 = v58;
          if ( !(_BYTE)v58 )
            goto LABEL_6;
        }
        else
        {
          if ( v53 == 1 )
          {
            v20 = (_QWORD *)v51[0];
            v37 = v51[1];
          }
          else
          {
            v20 = v51;
            v19 = 2;
          }
          v55.m128i_i64[0] = (__int64)v20;
          v21 = v58;
          BYTE1(v56) = v19;
          v22 = 2;
          v54 = (__m128i)(unsigned __int64)&v48;
          v55.m128i_i64[1] = v37;
          LOBYTE(v56) = 2;
          if ( !(_BYTE)v58 )
            goto LABEL_6;
        }
        if ( v21 == 1 )
        {
          v31 = _mm_loadu_si128(&v55);
          v22 = v56;
          v59 = _mm_loadu_si128(&v54);
          v61 = v56;
          v60 = v31;
          if ( !(_BYTE)v56 )
            goto LABEL_7;
        }
        else
        {
          if ( BYTE1(v56) == 1 )
          {
            v36 = v54.m128i_i64[1];
            v24 = (__m128i *)v54.m128i_i64[0];
          }
          else
          {
            v24 = &v54;
            v22 = 2;
          }
          if ( BYTE1(v58) == 1 )
          {
            v35 = v57[1];
            v25 = (_QWORD *)v57[0];
          }
          else
          {
            v25 = v57;
            v21 = 2;
          }
          v59.m128i_i64[0] = (__int64)v24;
          v60.m128i_i64[0] = (__int64)v25;
          v59.m128i_i64[1] = v36;
          LOBYTE(v61) = v22;
          v60.m128i_i64[1] = v35;
          BYTE1(v61) = v21;
        }
        v26 = v64;
        if ( (_BYTE)v64 )
        {
          if ( v22 == 1 )
          {
            v30 = _mm_loadu_si128(&v63);
            v65 = _mm_loadu_si128(&v62);
            v67 = v64;
            v66 = v30;
          }
          else if ( (_BYTE)v64 == 1 )
          {
            v32 = _mm_loadu_si128(&v60);
            v65 = _mm_loadu_si128(&v59);
            v67 = v61;
            v66 = v32;
          }
          else
          {
            if ( BYTE1(v61) == 1 )
            {
              v34 = v59.m128i_i64[1];
              v27 = (__m128i *)v59.m128i_i64[0];
            }
            else
            {
              v27 = &v59;
              v22 = 2;
            }
            if ( BYTE1(v64) == 1 )
            {
              v33 = v62.m128i_i64[1];
              v28 = (__m128i *)v62.m128i_i64[0];
            }
            else
            {
              v28 = &v62;
              v26 = 2;
            }
            v65.m128i_i64[0] = (__int64)v27;
            v66.m128i_i64[0] = (__int64)v28;
            v65.m128i_i64[1] = v34;
            LOBYTE(v67) = v22;
            v66.m128i_i64[1] = v33;
            BYTE1(v67) = v26;
          }
          goto LABEL_8;
        }
LABEL_7:
        LOWORD(v67) = 256;
LABEL_8:
        sub_11FD800(a1 + 176, a2, (__int64)&v65, 1);
        if ( (__int64 *)v40[0] != &v41 )
          j_j___libc_free_0(v40[0], v41 + 1);
        if ( (__int64 *)v42[0] != &v43 )
          j_j___libc_free_0(v42[0], v43 + 1);
        return 0;
      }
    }
    else
    {
      v45 = 256;
      v47 = 256;
      LOWORD(v50) = 256;
    }
    LOWORD(v56) = 256;
LABEL_6:
    LOWORD(v61) = 256;
    goto LABEL_7;
  }
  v15 = *(_BYTE *)(a3 + 32);
  if ( v15 )
  {
    if ( v15 == 1 )
    {
      v16 = (__m128i *)"'";
      v17 = 3;
      v6 = v62.m128i_i64[1];
      v62.m128i_i64[0] = (__int64)"'";
      LOWORD(v64) = 259;
    }
    else
    {
      if ( *(_BYTE *)(a3 + 33) == 1 )
      {
        v5 = *(_QWORD *)(a3 + 8);
        v10 = *(_QWORD *)a3;
      }
      else
      {
        v15 = 2;
      }
      v63.m128i_i64[0] = v10;
      v16 = &v62;
      v62.m128i_i64[0] = (__int64)"'";
      v63.m128i_i64[1] = v5;
      LOBYTE(v64) = 3;
      BYTE1(v64) = v15;
      v17 = 2;
    }
    v65.m128i_i64[0] = (__int64)v16;
    v65.m128i_i64[1] = v6;
    v66.m128i_i64[0] = (__int64)"' is not a basic block";
    LOBYTE(v67) = v17;
    BYTE1(v67) = 3;
  }
  else
  {
    LOWORD(v64) = 256;
    LOWORD(v67) = 256;
  }
  sub_11FD800(a1 + 176, a2, (__int64)&v65, 1);
  return 0;
}
