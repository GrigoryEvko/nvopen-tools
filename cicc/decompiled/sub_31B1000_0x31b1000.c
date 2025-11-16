// Function: sub_31B1000
// Address: 0x31b1000
//
__int64 __fastcall sub_31B1000(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // r13
  bool v7; // al
  __int64 *v8; // r14
  __int64 *v9; // rbx
  __int64 *v10; // rbx
  int v11; // r12d
  __int64 v12; // r13
  __int64 v13; // rdx
  int v14; // eax
  bool v15; // al
  __int64 *v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // r12
  _QWORD *i; // rax
  _QWORD *v22; // rax
  __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // r9
  int v29; // eax
  __m128i v30; // xmm1
  _QWORD *v31; // rax
  __int64 v32; // rax
  int v33; // eax
  __m128i v34; // xmm3
  __int64 v35; // rcx
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 v38; // r9
  __m128i v39; // xmm5
  _QWORD *v41; // rax
  __int64 *v42; // [rsp+0h] [rbp-150h]
  __int64 *v43; // [rsp+8h] [rbp-148h]
  __int64 v44; // [rsp+18h] [rbp-138h]
  __int64 v45; // [rsp+20h] [rbp-130h]
  __int64 v46; // [rsp+28h] [rbp-128h]
  _QWORD *v47; // [rsp+28h] [rbp-128h]
  __int64 v48; // [rsp+28h] [rbp-128h]
  __int64 v49; // [rsp+38h] [rbp-118h]
  unsigned int v50; // [rsp+40h] [rbp-110h]
  __int16 v51; // [rsp+46h] [rbp-10Ah]
  __int64 v52; // [rsp+48h] [rbp-108h]
  __m128i v53; // [rsp+50h] [rbp-100h] BYREF
  __int16 v54; // [rsp+60h] [rbp-F0h]
  __int64 v55; // [rsp+68h] [rbp-E8h]
  __int64 v56; // [rsp+70h] [rbp-E0h]
  __int64 v57; // [rsp+78h] [rbp-D8h]
  __int64 v58; // [rsp+80h] [rbp-D0h]
  __int64 v59; // [rsp+88h] [rbp-C8h]
  __m128i v60; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v61; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v62; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i v63; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v64; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v65; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v66; // [rsp+F0h] [rbp-60h] BYREF
  __m128i v67; // [rsp+100h] [rbp-50h]
  char v68; // [rsp+110h] [rbp-40h]
  char v69; // [rsp+111h] [rbp-3Fh]

  sub_31AFE90(&v53, a2, a3, a4);
  v5 = *a2;
  v52 = v53.m128i_i64[1];
  v6 = v53.m128i_i64[0];
  v51 = v54;
  v49 = v55;
  v7 = sub_318B630(*a2);
  if ( v5 && v7 && (*(_DWORD *)(v5 + 8) != 37 || sub_318B6C0(v5)) )
  {
    if ( sub_318B670(v5) )
    {
      v5 = sub_318B680(v5);
    }
    else if ( *(_DWORD *)(v5 + 8) == 37 )
    {
      v5 = sub_318B6C0(v5);
    }
  }
  v8 = sub_318EB80(v5);
  v9 = &a2[a3];
  v42 = v9;
  if ( (unsigned int)*(unsigned __int8 *)(*v8 + 8) - 17 > 1 )
  {
    if ( a2 == v9 )
    {
      v41 = sub_318E570((__int64)v8, 0);
      return sub_371B680(v41);
    }
    goto LABEL_9;
  }
  v8 = sub_318E560(v8);
  if ( a2 != v9 )
  {
LABEL_9:
    v46 = v6;
    v10 = a2;
    v11 = 0;
    do
    {
      v12 = *v10;
      v15 = sub_318B630(*v10);
      if ( v12 && v15 && (*(_DWORD *)(v12 + 8) != 37 || sub_318B6C0(v12)) )
      {
        if ( sub_318B670(v12) )
        {
          v12 = sub_318B680(v12);
        }
        else if ( *(_DWORD *)(v12 + 8) == 37 )
        {
          v12 = sub_318B6C0(v12);
        }
      }
      v13 = *sub_318EB80(v12);
      v14 = 1;
      if ( *(_BYTE *)(v13 + 8) == 17 )
        v14 = *(_DWORD *)(v13 + 32);
      ++v10;
      v11 += v14;
    }
    while ( v42 != v10 );
    v6 = v46;
    goto LABEL_22;
  }
  v11 = 0;
LABEL_22:
  if ( (unsigned int)*(unsigned __int8 *)(*v8 + 8) - 17 <= 1 )
  {
    v16 = sub_318E560(v8);
    v17 = *v8;
    v8 = v16;
    v11 *= *(_DWORD *)(v17 + 32);
  }
  v18 = sub_318E570((__int64)v8, v11);
  v19 = (_QWORD *)sub_371B680(v18);
  v45 = *a2;
  v20 = *(_QWORD *)(*a2 + 24);
  if ( a2 != v42 )
  {
    v50 = 0;
    v43 = a2 + 1;
    for ( i = sub_318EB80(v45); ; i = sub_318EB80(v35) )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*i + 8LL) - 17 > 1 )
      {
        v36 = sub_318E530(v20);
        v37 = sub_371B650(v36, v50);
        v56 = v6;
        v66.m128i_i64[0] = (__int64)"Pack";
        v69 = 1;
        v57 = v52;
        v68 = 3;
        LOWORD(v58) = v51;
        v59 = v49;
        v19 = sub_318BC00((__int64)v19, v45, v37, v20, (__int64)&v66, v38, v6, v52, v58);
        ++v50;
        if ( sub_318B630((__int64)v19) && v19 )
        {
          sub_318B480((__int64)&v64, (__int64)v19);
          v39 = _mm_loadu_si128(&v65);
          v66 = _mm_loadu_si128(&v64);
          v67 = v39;
          sub_371B2F0(&v66);
          v52 = v66.m128i_i64[1];
          v6 = v66.m128i_i64[0];
          v51 = v67.m128i_i16[0];
          v49 = v67.m128i_i64[1];
        }
      }
      else
      {
        v22 = sub_318EB80(v45);
        v44 = *(int *)(*v22 + 32LL);
        if ( *(_DWORD *)(*v22 + 32LL) )
        {
          v23 = 0;
          do
          {
            v24 = sub_318E530(v20);
            v25 = sub_371B650(v24, (int)v23);
            v69 = 1;
            v56 = v6;
            v66.m128i_i64[0] = (__int64)"VPack";
            v68 = 3;
            v57 = v52;
            LOWORD(v58) = v51;
            v59 = v49;
            v28 = sub_318BDB0(v45, v25, v20, (__int64)&v66, v26, v27, v6, v52, v58);
            v29 = *((_DWORD *)v28 + 2);
            if ( v29 && (unsigned int)(v29 - 5) > 0x12 )
            {
              v47 = v28;
              sub_318B480((__int64)&v60, (__int64)v28);
              v30 = _mm_loadu_si128(&v61);
              v66 = _mm_loadu_si128(&v60);
              v67 = v30;
              sub_371B2F0(&v66);
              v6 = v66.m128i_i64[0];
              v28 = v47;
              v52 = v66.m128i_i64[1];
              v51 = v67.m128i_i16[0];
              v49 = v67.m128i_i64[1];
            }
            v48 = (__int64)v28;
            v31 = sub_318E530(v20);
            v32 = sub_371B650(v31, (unsigned int)v23 + v50);
            v56 = v6;
            v66.m128i_i64[0] = (__int64)"VPack";
            v69 = 1;
            v57 = v52;
            v68 = 3;
            LOWORD(v58) = v51;
            v59 = v49;
            v19 = sub_318BC00((__int64)v19, v48, v32, v20, (__int64)&v66, v48, v6, v52, v58);
            v33 = *((_DWORD *)v19 + 2);
            if ( v33 )
            {
              if ( (unsigned int)(v33 - 5) > 0x12 )
              {
                sub_318B480((__int64)&v62, (__int64)v19);
                v34 = _mm_loadu_si128(&v63);
                v66 = _mm_loadu_si128(&v62);
                v67 = v34;
                sub_371B2F0(&v66);
                v52 = v66.m128i_i64[1];
                v6 = v66.m128i_i64[0];
                v51 = v67.m128i_i16[0];
                v49 = v67.m128i_i64[1];
              }
            }
            ++v23;
          }
          while ( v44 != v23 );
          v50 += v44;
        }
      }
      if ( v43 == v42 )
        break;
      v35 = *v43++;
      v45 = v35;
    }
  }
  return (__int64)v19;
}
