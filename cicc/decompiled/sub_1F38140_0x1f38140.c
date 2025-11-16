// Function: sub_1F38140
// Address: 0x1f38140
//
void __fastcall sub_1F38140(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  _QWORD *v6; // r15
  __int64 *v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 *v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rax
  int v16; // ecx
  int v17; // r9d
  _QWORD *v18; // r10
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // r13
  __m128i v24; // xmm0
  _QWORD *v25; // rsi
  _QWORD *v26; // rdi
  __int64 v27; // rdx
  _WORD *v28; // rdx
  _QWORD *v29; // rsi
  _QWORD *v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r13
  __m128i *v35; // rax
  __m128i v36; // xmm0
  _BYTE *v37; // rax
  __int64 v38; // r13
  __m128i *v39; // rax
  __m128i v40; // xmm0
  _WORD *v41; // rdx
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  _QWORD *v44; // rsi
  __int64 v45; // r13
  void *v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rax
  __m128i *v49; // rdx
  __int64 v50; // r13
  __m128i si128; // xmm0
  _WORD *v52; // rdx
  _QWORD *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rcx
  _QWORD *v56; // rdi
  __int64 v57; // r13
  __m128i *v58; // rax
  __m128i v59; // xmm0
  _BYTE *v60; // rax
  __int64 v61; // [rsp-10h] [rbp-1F0h]
  _QWORD *v62; // [rsp-10h] [rbp-1F0h]
  _QWORD *v63; // [rsp-8h] [rbp-1E8h]
  _QWORD *v64; // [rsp+10h] [rbp-1D0h]
  int i; // [rsp+24h] [rbp-1BCh]
  _QWORD v66[2]; // [rsp+50h] [rbp-190h] BYREF
  void (__fastcall *v67)(_QWORD *, _QWORD *, __int64, __int64); // [rsp+60h] [rbp-180h]
  void (__fastcall *v68)(_QWORD *, __int64); // [rsp+68h] [rbp-178h]
  _QWORD v69[2]; // [rsp+70h] [rbp-170h] BYREF
  void (__fastcall *v70)(_QWORD *, _QWORD *, __int64); // [rsp+80h] [rbp-160h]
  void (__fastcall *v71)(_QWORD *, __int64); // [rsp+88h] [rbp-158h]
  _QWORD v72[2]; // [rsp+90h] [rbp-150h] BYREF
  void (__fastcall *v73)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // [rsp+A0h] [rbp-140h]
  void (__fastcall *v74)(_QWORD *, __int64); // [rsp+A8h] [rbp-138h]
  _QWORD v75[2]; // [rsp+B0h] [rbp-130h] BYREF
  void (__fastcall *v76)(_QWORD *, _QWORD *, __int64); // [rsp+C0h] [rbp-120h]
  void (__fastcall *v77)(_QWORD *, __int64); // [rsp+C8h] [rbp-118h]
  _QWORD v78[2]; // [rsp+D0h] [rbp-110h] BYREF
  void (__fastcall *v79)(_QWORD *, _QWORD *, __int64); // [rsp+E0h] [rbp-100h]
  void (__fastcall *v80)(_QWORD *, __int64); // [rsp+E8h] [rbp-F8h]
  _QWORD v81[2]; // [rsp+F0h] [rbp-F0h] BYREF
  void (__fastcall *v82)(_QWORD *, _QWORD *, __int64); // [rsp+100h] [rbp-E0h]
  void (__fastcall *v83)(_QWORD *, __int64); // [rsp+108h] [rbp-D8h]
  __int64 v84; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v85; // [rsp+118h] [rbp-C8h]
  _QWORD *v86; // [rsp+120h] [rbp-C0h] BYREF
  int v87; // [rsp+128h] [rbp-B8h]
  __int64 *v88; // [rsp+160h] [rbp-80h] BYREF
  __int64 v89; // [rsp+168h] [rbp-78h]
  _BYTE v90[112]; // [rsp+170h] [rbp-70h] BYREF

  v64 = (_QWORD *)(a1 + 320);
  v6 = *(_QWORD **)(*(_QWORD *)(a1 + 328) + 8LL);
  if ( (_QWORD *)(a1 + 320) != v6 )
  {
    do
    {
      v84 = 0;
      v8 = (__int64 *)v6[9];
      v85 = 1;
      v9 = v6[8];
      v10 = (unsigned __int64 *)&v86;
      do
        *v10++ = -8;
      while ( v10 != (unsigned __int64 *)&v88 );
      v11 = &v84;
      v88 = (__int64 *)v90;
      v89 = 0x800000000LL;
      sub_1F36F80((__int64)&v84, (__int64 *)v9, v8, a4, a5, a6);
      v12 = v6[4];
      if ( (_QWORD *)v12 != v6 + 3 )
      {
        while ( 1 )
        {
          if ( **(_WORD **)(v12 + 16) != 45 && **(_WORD **)(v12 + 16) )
            goto LABEL_7;
          v13 = *(unsigned int *)(v12 + 40);
          a5 = (__int64)&v88[(unsigned int)v89];
          if ( v88 != (__int64 *)a5 )
            break;
          if ( (_DWORD)v13 != 1 )
          {
            v15 = *(_QWORD *)(v12 + 32);
            goto LABEL_20;
          }
LABEL_46:
          if ( (*(_BYTE *)v12 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
              v12 = *(_QWORD *)(v12 + 8);
          }
          v12 = *(_QWORD *)(v12 + 8);
          if ( v6 + 3 == (_QWORD *)v12 )
            goto LABEL_7;
        }
        v11 = v88;
        do
        {
          v14 = *v11;
          if ( (_DWORD)v13 == 1 )
          {
LABEL_73:
            v48 = sub_16BA580((__int64)v11, v9, v13);
            v49 = *(__m128i **)(v48 + 24);
            v50 = v48;
            if ( *(_QWORD *)(v48 + 16) - (_QWORD)v49 <= 0x10u )
            {
              v50 = sub_16E7EE0(v48, "Malformed PHI in ", 0x11u);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_42F1A70);
              v49[1].m128i_i8[0] = 32;
              *v49 = si128;
              *(_QWORD *)(v48 + 24) += 17LL;
            }
            v25 = v6;
            v26 = v66;
            sub_1DD5B60(v66, (__int64)v6);
            if ( v67 )
            {
              v68(v66, v50);
              v52 = *(_WORD **)(v50 + 24);
              if ( *(_QWORD *)(v50 + 16) - (_QWORD)v52 <= 1u )
              {
                v50 = sub_16E7EE0(v50, ": ", 2u);
              }
              else
              {
                *v52 = 8250;
                *(_QWORD *)(v50 + 24) += 2LL;
              }
              v53 = (_QWORD *)v50;
              sub_1E1A330(v12, v50, 1, 0, 0, 1, 0);
              v56 = v62;
              if ( v67 )
              {
                v53 = v66;
                v56 = v66;
                v67(v66, v66, 3, v55);
              }
              v57 = sub_16BA580((__int64)v56, (__int64)v53, v54);
              v58 = *(__m128i **)(v57 + 24);
              if ( *(_QWORD *)(v57 + 16) - (_QWORD)v58 <= 0x20u )
              {
                v57 = sub_16E7EE0(v57, "  missing input from predecessor ", 0x21u);
              }
              else
              {
                v59 = _mm_load_si128((const __m128i *)&xmmword_42F1A80);
                v58[2].m128i_i8[0] = 32;
                *v58 = v59;
                v58[1] = _mm_load_si128((const __m128i *)&xmmword_42F1A90);
                *(_QWORD *)(v57 + 24) += 33LL;
              }
              v25 = (_QWORD *)v14;
              v26 = v69;
              sub_1DD5B60(v69, v14);
              if ( v70 )
              {
                v71(v69, v57);
                v60 = *(_BYTE **)(v57 + 24);
                if ( (unsigned __int64)v60 >= *(_QWORD *)(v57 + 16) )
                {
                  sub_16E7DE0(v57, 10);
                }
                else
                {
                  *(_QWORD *)(v57 + 24) = v60 + 1;
                  *v60 = 10;
                }
                if ( v70 )
                  v70(v69, v69, 3);
LABEL_97:
                BUG();
              }
            }
LABEL_92:
            sub_4263D6(v26, v25, v27);
          }
          v15 = *(_QWORD *)(v12 + 32);
          v16 = 1;
          while ( 1 )
          {
            v9 = 5LL * (unsigned int)(v16 + 1);
            if ( v14 == *(_QWORD *)(v15 + 40LL * (unsigned int)(v16 + 1) + 24) )
              break;
            v16 += 2;
            if ( v16 == (_DWORD)v13 )
              goto LABEL_73;
          }
          ++v11;
        }
        while ( (__int64 *)a5 != v11 );
LABEL_20:
        v17 = 8;
        LODWORD(a4) = 1;
        v18 = &v86;
        a5 = v85 & 1;
        if ( (v85 & 1) == 0 )
        {
          v18 = v86;
          v17 = v87;
        }
        a6 = (__int64 *)(unsigned int)(v17 - 1);
        while ( 1 )
        {
          v19 = 5LL * (unsigned int)(a4 + 1);
          v20 = *(_QWORD *)(v15 + 40LL * (unsigned int)(a4 + 1) + 24);
          if ( !a2 )
            goto LABEL_23;
          if ( !(_BYTE)a5 && !v87 )
          {
LABEL_31:
            v21 = sub_16BA580((__int64)v11, v19, v13);
            v22 = *(__m128i **)(v21 + 24);
            v23 = v21;
            if ( *(_QWORD *)(v21 + 16) - (_QWORD)v22 <= 0x19u )
            {
              v23 = sub_16E7EE0(v21, "Warning: malformed PHI in ", 0x1Au);
            }
            else
            {
              v24 = _mm_load_si128((const __m128i *)&xmmword_42F1AA0);
              qmemcpy(&v22[1], "ed PHI in ", 10);
              *v22 = v24;
              *(_QWORD *)(v21 + 24) += 26LL;
            }
            v25 = v6;
            v26 = v72;
            sub_1DD5B60(v72, (__int64)v6);
            if ( v73 )
            {
              v74(v72, v23);
              v28 = *(_WORD **)(v23 + 24);
              if ( *(_QWORD *)(v23 + 16) - (_QWORD)v28 <= 1u )
              {
                v23 = sub_16E7EE0(v23, ": ", 2u);
              }
              else
              {
                *v28 = 8250;
                *(_QWORD *)(v23 + 24) += 2LL;
              }
              v29 = (_QWORD *)v23;
              v30 = (_QWORD *)v12;
              sub_1E1A330(v12, v23, 1, 0, 0, 1, 0);
              if ( v73 )
              {
                v29 = v72;
                v30 = v72;
                v73(v72, v72, 3, v32, v33, v61);
              }
              v34 = sub_16BA580((__int64)v30, (__int64)v29, v31);
              v35 = *(__m128i **)(v34 + 24);
              if ( *(_QWORD *)(v34 + 16) - (_QWORD)v35 <= 0x1Eu )
              {
                v34 = sub_16E7EE0(v34, "  extra input from predecessor ", 0x1Fu);
              }
              else
              {
                v36 = _mm_load_si128((const __m128i *)&xmmword_42F1AB0);
                qmemcpy(&v35[1], "om predecessor ", 15);
                *v35 = v36;
                *(_QWORD *)(v34 + 24) += 31LL;
              }
              v25 = (_QWORD *)v20;
              v26 = v75;
              sub_1DD5B60(v75, v20);
              if ( v76 )
              {
                v77(v75, v34);
                v37 = *(_BYTE **)(v34 + 24);
                if ( (unsigned __int64)v37 >= *(_QWORD *)(v34 + 16) )
                {
                  sub_16E7DE0(v34, 10);
                }
                else
                {
                  *(_QWORD *)(v34 + 24) = v37 + 1;
                  *v37 = 10;
                }
                if ( v76 )
                  v76(v75, v75, 3);
                goto LABEL_97;
              }
            }
            goto LABEL_92;
          }
          v19 = (unsigned int)a6 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v11 = (__int64 *)v18[v19];
          if ( v11 == (__int64 *)v20 )
          {
LABEL_23:
            v9 = *(unsigned int *)(v20 + 48);
            if ( (int)v9 < 0 )
              goto LABEL_54;
          }
          else
          {
            for ( i = 1; ; ++i )
            {
              if ( v11 == (__int64 *)-8LL )
                goto LABEL_31;
              v19 = (unsigned int)a6 & (i + (_DWORD)v19);
              v11 = (__int64 *)v18[(unsigned int)v19];
              if ( (__int64 *)v20 == v11 )
                break;
            }
            v9 = *(unsigned int *)(v20 + 48);
            if ( (int)v9 < 0 )
            {
LABEL_54:
              v38 = sub_16BA580((__int64)v11, v9, v13);
              v39 = *(__m128i **)(v38 + 24);
              if ( *(_QWORD *)(v38 + 16) - (_QWORD)v39 <= 0x10u )
              {
                v38 = sub_16E7EE0(v38, "Malformed PHI in ", 0x11u);
              }
              else
              {
                v40 = _mm_load_si128((const __m128i *)&xmmword_42F1A70);
                v39[1].m128i_i8[0] = 32;
                *v39 = v40;
                *(_QWORD *)(v38 + 24) += 17LL;
              }
              v25 = v6;
              v26 = v78;
              sub_1DD5B60(v78, (__int64)v6);
              if ( v79 )
              {
                v80(v78, v38);
                v41 = *(_WORD **)(v38 + 24);
                if ( *(_QWORD *)(v38 + 16) - (_QWORD)v41 <= 1u )
                {
                  v38 = sub_16E7EE0(v38, ": ", 2u);
                }
                else
                {
                  *v41 = 8250;
                  *(_QWORD *)(v38 + 24) += 2LL;
                }
                v42 = (_QWORD *)v12;
                sub_1E1A330(v12, v38, 1, 0, 0, 1, 0);
                v44 = v63;
                if ( v79 )
                {
                  v44 = v78;
                  v42 = v78;
                  v79(v78, v78, 3);
                }
                v45 = sub_16BA580((__int64)v42, (__int64)v44, v43);
                v46 = *(void **)(v45 + 24);
                if ( *(_QWORD *)(v45 + 16) - (_QWORD)v46 <= 0xEu )
                {
                  v45 = sub_16E7EE0(v45, "  non-existing ", 0xFu);
                }
                else
                {
                  qmemcpy(v46, "  non-existing ", 15);
                  *(_QWORD *)(v45 + 24) += 15LL;
                }
                v25 = (_QWORD *)v20;
                v26 = v81;
                sub_1DD5B60(v81, v20);
                if ( v82 )
                {
                  v83(v81, v45);
                  v47 = *(_BYTE **)(v45 + 24);
                  if ( (unsigned __int64)v47 >= *(_QWORD *)(v45 + 16) )
                  {
                    sub_16E7DE0(v45, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v45 + 24) = v47 + 1;
                    *v47 = 10;
                  }
                  if ( v82 )
                    v82(v81, v81, 3);
                  goto LABEL_97;
                }
              }
              goto LABEL_92;
            }
          }
          a4 = (unsigned int)(a4 + 2);
          if ( (_DWORD)a4 == (_DWORD)v13 )
            goto LABEL_46;
        }
      }
LABEL_7:
      if ( v88 != (__int64 *)v90 )
        _libc_free((unsigned __int64)v88);
      if ( (v85 & 1) == 0 )
        j___libc_free_0(v86);
      v6 = (_QWORD *)v6[1];
    }
    while ( v64 != v6 );
  }
}
