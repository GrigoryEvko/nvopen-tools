// Function: sub_FCCAD0
// Address: 0xfccad0
//
__int64 __fastcall sub_FCCAD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  const __m128i *v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int32 v15; // edi
  __int64 v16; // r12
  int v17; // esi
  __int64 result; // rax
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // r13
  __int64 v29; // r11
  __int64 v30; // rdx
  int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // r8
  unsigned __int64 v34; // r14
  __int64 *v35; // rax
  __int64 v36; // r10
  signed __int64 v37; // r13
  const void *v38; // r11
  unsigned __int64 v39; // r9
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  _QWORD *v43; // rax
  _QWORD *i; // r10
  __int64 *v45; // r13
  unsigned int v46; // r12d
  unsigned __int8 *v47; // r14
  int v48; // ebx
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r12
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r12
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // [rsp+8h] [rbp-198h]
  __int64 v69; // [rsp+10h] [rbp-190h]
  __int64 v70; // [rsp+10h] [rbp-190h]
  const void *v71; // [rsp+10h] [rbp-190h]
  __int64 v72; // [rsp+18h] [rbp-188h]
  __int64 v73; // [rsp+18h] [rbp-188h]
  unsigned __int64 v74; // [rsp+30h] [rbp-170h]
  __int64 v75; // [rsp+30h] [rbp-170h]
  __int64 **v76; // [rsp+38h] [rbp-168h]
  __int64 v77; // [rsp+40h] [rbp-160h]
  __int64 v78; // [rsp+40h] [rbp-160h]
  __int64 *v79; // [rsp+40h] [rbp-160h]
  __int64 v80; // [rsp+40h] [rbp-160h]
  unsigned int v81; // [rsp+4Ch] [rbp-154h]
  __int64 v82; // [rsp+70h] [rbp-130h] BYREF
  __int64 v83; // [rsp+78h] [rbp-128h]
  __int64 v84; // [rsp+80h] [rbp-120h]
  __int64 *v85; // [rsp+90h] [rbp-110h] BYREF
  __int64 v86; // [rsp+98h] [rbp-108h]
  _BYTE dest[64]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 *v88; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+E8h] [rbp-B8h]
  _BYTE v90[176]; // [rsp+F0h] [rbp-B0h] BYREF

  v7 = *(_DWORD *)(a1 + 80);
  if ( v7 )
  {
    while ( 1 )
    {
      v9 = v7;
      v10 = v7 - 1;
      v11 = (const __m128i *)(*(_QWORD *)(a1 + 72) + 24 * v9 - 24);
      v12 = _mm_loadu_si128(v11);
      v13 = v11[1].m128i_i64[0];
      v14 = v11->m128i_i8[0] & 3;
      v15 = v11->m128i_i32[1];
      v16 = v11->m128i_i64[1];
      v17 = ((unsigned __int32)v11->m128i_i32[0] >> 2) & 0x1FFFFFFF;
      *(_DWORD *)(a1 + 80) = v10;
      *(_DWORD *)(a1 + 16) = v17;
      if ( (_BYTE)v14 == 2 )
        break;
      if ( (_BYTE)v14 == 3 )
      {
        sub_FCC8E0(a1, v16, v10, (__int64 *)v14, a5, a6);
        v7 = *(_DWORD *)(a1 + 80);
        if ( !v7 )
          goto LABEL_9;
      }
      else
      {
        if ( (_BYTE)v14 == 1 )
        {
          v28 = *(unsigned int *)(a1 + 224);
          v29 = *(_QWORD *)(a1 + 216);
          v30 = 0;
          v86 = 0x800000000LL;
          v31 = v28;
          v28 *= 8;
          v32 = v31 - v15;
          v33 = v29 + v28;
          v34 = v32;
          v81 = v32;
          v35 = (__int64 *)dest;
          v36 = 8 * v34;
          v37 = v28 - 8 * v34;
          v85 = (__int64 *)dest;
          v38 = (const void *)(8 * v34 + v29);
          v39 = v37 >> 3;
          if ( (unsigned __int64)v37 > 0x40 )
          {
            v71 = v38;
            v75 = v33;
            sub_C8D5F0((__int64)&v85, dest, v37 >> 3, 8u, v33, v39);
            v30 = (unsigned int)v86;
            v38 = v71;
            v36 = 8 * v34;
            v33 = v75;
            v39 = v37 >> 3;
            v35 = &v85[(unsigned int)v86];
          }
          if ( (const void *)v33 != v38 )
          {
            v74 = v39;
            v77 = v36;
            memcpy(v35, v38, v37);
            v30 = (unsigned int)v86;
            v39 = v74;
            v36 = v77;
          }
          v40 = *(unsigned int *)(a1 + 224);
          v41 = v39 + v30;
          LODWORD(v86) = v41;
          v42 = (unsigned int)v41;
          if ( v34 != v40 )
          {
            if ( v34 < v40 )
            {
              *(_DWORD *)(a1 + 224) = v81;
            }
            else
            {
              if ( v34 > *(unsigned int *)(a1 + 228) )
              {
                v80 = v36;
                sub_C8D5F0(a1 + 216, (const void *)(a1 + 232), v34, 8u, v33, v39);
                v40 = *(unsigned int *)(a1 + 224);
                v36 = v80;
              }
              v41 = *(_QWORD *)(a1 + 216);
              v43 = (_QWORD *)(v41 + 8 * v40);
              for ( i = (_QWORD *)(v41 + v36); i != v43; ++v43 )
              {
                if ( v43 )
                  *v43 = 0;
              }
              v42 = (unsigned int)v86;
              *(_DWORD *)(a1 + 224) = v81;
            }
          }
          v45 = v85;
          v78 = (unsigned int)v42;
          v88 = (__int64 *)v90;
          v89 = 0x1000000000LL;
          if ( v13 && (unsigned int)*(_QWORD *)(*(_QWORD *)(v13 + 8) + 32LL) )
          {
            v72 = v16;
            v46 = 0;
            v47 = (unsigned __int8 *)v13;
            v48 = *(_QWORD *)(*(_QWORD *)(v13 + 8) + 32LL);
            do
            {
              v49 = sub_AD69F0(v47, v46);
              v41 = (unsigned int)v89;
              v39 = (unsigned int)v89 + 1LL;
              if ( v39 > HIDWORD(v89) )
              {
                v70 = v49;
                sub_C8D5F0((__int64)&v88, v90, (unsigned int)v89 + 1LL, 8u, v33, v39);
                v41 = (unsigned int)v89;
                v49 = v70;
              }
              v42 = (__int64)v88;
              ++v46;
              v88[v41] = v49;
              LODWORD(v89) = v89 + 1;
            }
            while ( v48 != v46 );
            v16 = v72;
          }
          if ( v12.m128i_i8[3] < 0 )
          {
            v64 = (__int64 *)sub_BD5C60(v16);
            v68 = sub_BCE3C0(v64, 0);
            v65 = *(__int64 **)(*(_QWORD *)(*v45 + 8) + 16LL);
            v82 = *v65;
            v66 = v65[1];
            v84 = v68;
            v83 = v66;
            v67 = (_QWORD *)sub_BD5C60(v16);
            v76 = (__int64 **)sub_BD0B90(v67, &v82, 3, 0);
          }
          v79 = &v45[v78];
          if ( v45 == v79 )
          {
            v41 = (unsigned int)v89;
          }
          else
          {
            v73 = v16;
            do
            {
              v60 = *v45;
              if ( v12.m128i_i8[3] < 0 )
              {
                v50 = sub_FC8800(a1, *(_QWORD *)(v60 - 32LL * (*(_DWORD *)(v60 + 4) & 0x7FFFFFF)), v41, v42, v33, v39);
                v51 = *(_DWORD *)(v60 + 4) & 0x7FFFFFF;
                v52 = *(_QWORD *)(v60 + 32 * (1 - v51));
                v56 = sub_FC8800(a1, v52, v51, v53, v54, v55);
                v57 = sub_AD6530(v68, v52);
                v82 = v50;
                v83 = v56;
                v84 = v57;
                v58 = sub_AD24A0(v76, &v82, 3);
              }
              else
              {
                v58 = sub_FC8800(a1, *v45, v41, v42, v33, v39);
              }
              v59 = (unsigned int)v89;
              v39 = (unsigned int)v89 + 1LL;
              if ( v39 > HIDWORD(v89) )
              {
                v69 = v58;
                sub_C8D5F0((__int64)&v88, v90, (unsigned int)v89 + 1LL, 8u, v33, v39);
                v59 = (unsigned int)v89;
                v58 = v69;
              }
              v42 = (__int64)v88;
              ++v45;
              v88[v59] = v58;
              v41 = (unsigned int)(v89 + 1);
              LODWORD(v89) = v89 + 1;
            }
            while ( v79 != v45 );
            v16 = v73;
          }
          v61 = sub_AD1300(*(__int64 ***)(v16 + 24), v88, v41);
          sub_B30160(v16, v61);
          if ( v88 != (__int64 *)v90 )
            _libc_free(v88, v61);
          if ( v85 != (__int64 *)dest )
            _libc_free(v85, v61);
        }
        else
        {
          v8 = sub_FC8800(a1, v13, v10, v14, a5, a6);
          sub_B30160(v16, v8);
          sub_FCBCE0(a1, v16);
        }
LABEL_5:
        v7 = *(_DWORD *)(a1 + 80);
        if ( !v7 )
          goto LABEL_9;
      }
    }
    v27 = sub_FC8800(a1, v13, v10, v14, a5, a6);
    if ( *(_BYTE *)v16 == 1 )
    {
      sub_B303B0(v16, v27);
    }
    else
    {
      if ( *(_BYTE *)v16 != 2 )
        BUG();
      if ( *(_QWORD *)(v16 - 32) )
      {
        v62 = *(_QWORD *)(v16 - 24);
        **(_QWORD **)(v16 - 16) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = *(_QWORD *)(v16 - 16);
      }
      *(_QWORD *)(v16 - 32) = v27;
      if ( v27 )
      {
        v63 = *(_QWORD *)(v27 + 16);
        *(_QWORD *)(v16 - 24) = v63;
        if ( v63 )
        {
          a4 = v16 - 24;
          *(_QWORD *)(v63 + 16) = v16 - 24;
        }
        *(_QWORD *)(v16 - 16) = v27 + 16;
        *(_QWORD *)(v27 + 16) = v16 - 32;
      }
    }
    goto LABEL_5;
  }
LABEL_9:
  result = *(unsigned int *)(a1 + 192);
  for ( *(_DWORD *)(a1 + 16) = 0; (_DWORD)result; result = *(unsigned int *)(a1 + 192) )
  {
    v19 = *(_QWORD *)(a1 + 184);
    v20 = (__int64 *)(v19 + 16 * result - 16);
    v21 = v20[1];
    v22 = *v20;
    v20[1] = 0;
    v23 = (unsigned int)(*(_DWORD *)(a1 + 192) - 1);
    *(_DWORD *)(a1 + 192) = v23;
    v24 = *(_QWORD *)(a1 + 184) + 16 * v23;
    v25 = *(_QWORD *)(v24 + 8);
    if ( v25 )
    {
      sub_AA5290(*(_QWORD *)(v24 + 8));
      j_j___libc_free_0(v25, 80);
    }
    v26 = sub_FC8800(a1, v22, v19, a4, a5, a6);
    if ( !v26 )
      v26 = v22;
    sub_BD84D0(v21, v26);
    if ( v21 )
    {
      sub_AA5290(v21);
      j_j___libc_free_0(v21, 80);
    }
  }
  return result;
}
