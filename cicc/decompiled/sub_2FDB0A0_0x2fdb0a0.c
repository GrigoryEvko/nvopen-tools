// Function: sub_2FDB0A0
// Address: 0x2fdb0a0
//
__int64 __fastcall sub_2FDB0A0(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rsi
  __int64 *v8; // r13
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 v11; // r8
  int v12; // edx
  __int64 *v13; // r9
  __int64 *v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rax
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // r11
  __int64 *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // r10
  __int64 v24; // r13
  __m128i *v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rsi
  _QWORD *v28; // rdi
  __int64 v29; // rdx
  _WORD *v30; // rdx
  _QWORD *v31; // rsi
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r13
  __m128i *v36; // rax
  __m128i v37; // xmm0
  _BYTE *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r13
  __m128i *v41; // rax
  __m128i v42; // xmm0
  _WORD *v43; // rdx
  _QWORD *v44; // rdi
  _QWORD *v45; // rsi
  __int64 v46; // r13
  void *v47; // rax
  _BYTE *v48; // rax
  __int64 v49; // rax
  __m128i *v50; // rdx
  __int64 v51; // r13
  __m128i si128; // xmm0
  _WORD *v53; // rdx
  _QWORD *v54; // rsi
  _QWORD *v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r13
  __m128i *v59; // rax
  __m128i v60; // xmm0
  _BYTE *v61; // rax
  __int64 v62; // [rsp-10h] [rbp-1B0h]
  __int64 v63; // [rsp-10h] [rbp-1B0h]
  _QWORD *v64; // [rsp-8h] [rbp-1A8h]
  __int64 v65; // [rsp+8h] [rbp-198h]
  int i; // [rsp+14h] [rbp-18Ch]
  __int64 v67; // [rsp+18h] [rbp-188h]
  __int64 v68; // [rsp+30h] [rbp-170h]
  __int64 v69; // [rsp+38h] [rbp-168h]
  _QWORD v70[2]; // [rsp+40h] [rbp-160h] BYREF
  void (__fastcall *v71)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // [rsp+50h] [rbp-150h]
  void (__fastcall *v72)(_QWORD *, __int64); // [rsp+58h] [rbp-148h]
  _QWORD v73[2]; // [rsp+60h] [rbp-140h] BYREF
  void (__fastcall *v74)(_QWORD *, _QWORD *, __int64); // [rsp+70h] [rbp-130h]
  void (__fastcall *v75)(_QWORD *, __int64); // [rsp+78h] [rbp-128h]
  _QWORD v76[2]; // [rsp+80h] [rbp-120h] BYREF
  void (__fastcall *v77)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // [rsp+90h] [rbp-110h]
  void (__fastcall *v78)(_QWORD *, __int64); // [rsp+98h] [rbp-108h]
  _QWORD v79[2]; // [rsp+A0h] [rbp-100h] BYREF
  void (__fastcall *v80)(_QWORD *, _QWORD *, __int64); // [rsp+B0h] [rbp-F0h]
  void (__fastcall *v81)(_QWORD *, __int64); // [rsp+B8h] [rbp-E8h]
  _QWORD v82[2]; // [rsp+C0h] [rbp-E0h] BYREF
  void (__fastcall *v83)(_QWORD *, _QWORD *, __int64); // [rsp+D0h] [rbp-D0h]
  void (__fastcall *v84)(_QWORD *, __int64); // [rsp+D8h] [rbp-C8h]
  _QWORD v85[2]; // [rsp+E0h] [rbp-C0h] BYREF
  void (__fastcall *v86)(_QWORD *, _QWORD *, __int64); // [rsp+F0h] [rbp-B0h]
  void (__fastcall *v87)(_QWORD *, __int64); // [rsp+F8h] [rbp-A8h]
  __int64 v88; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v89; // [rsp+108h] [rbp-98h]
  __int64 v90; // [rsp+110h] [rbp-90h]
  __int64 v91; // [rsp+118h] [rbp-88h]
  __int64 *v92; // [rsp+120h] [rbp-80h]
  __int64 v93; // [rsp+128h] [rbp-78h]
  _BYTE v94[112]; // [rsp+130h] [rbp-70h] BYREF

  v65 = a1 + 320;
  result = *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8LL);
  v69 = result;
  if ( a1 + 320 != result )
  {
    do
    {
      v88 = 0;
      v89 = 0;
      v4 = *(unsigned int *)(v69 + 72);
      v5 = *(_QWORD *)(v69 + 64);
      v90 = 0;
      v91 = 0;
      v6 = v5 + 8 * v4;
      v92 = (__int64 *)v94;
      v93 = 0x800000000LL;
      v68 = v69 + 48;
      if ( v6 == v5 )
      {
        v8 = (__int64 *)v94;
        v10 = 0;
        v7 = 0;
        v9 = *(_QWORD *)(v69 + 56);
        if ( v68 != v9 )
          goto LABEL_5;
      }
      else
      {
        do
        {
          v7 = v5;
          v5 += 8;
          sub_2FD92E0((__int64)&v88, (__int64 *)v7);
        }
        while ( v6 != v5 );
        v8 = v92;
        v9 = *(_QWORD *)(v69 + 56);
        if ( v9 != v68 )
        {
          while ( 1 )
          {
LABEL_5:
            if ( *(_WORD *)(v9 + 68) && *(_WORD *)(v9 + 68) != 68 )
              goto LABEL_7;
            v11 = (unsigned int)v93;
            v12 = *(_DWORD *)(v9 + 40) & 0xFFFFFF;
            v13 = &v8[v11];
            if ( &v8[v11] != v8 )
              break;
            if ( v12 != 1 )
            {
              v16 = *(_QWORD *)(v9 + 32);
              goto LABEL_19;
            }
LABEL_32:
            if ( (*(_BYTE *)v9 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
                v9 = *(_QWORD *)(v9 + 8);
            }
            v9 = *(_QWORD *)(v9 + 8);
            if ( v68 == v9 )
              goto LABEL_7;
          }
          v14 = v8;
          do
          {
            v15 = *v14;
            if ( v12 == 1 )
            {
LABEL_93:
              v49 = sub_C5F790((__int64)v14, v7);
              v50 = *(__m128i **)(v49 + 32);
              v51 = v49;
              if ( *(_QWORD *)(v49 + 24) - (_QWORD)v50 <= 0x10u )
              {
                v51 = sub_CB6200(v49, "Malformed PHI in ", 0x11u);
              }
              else
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_42F1A70);
                v50[1].m128i_i8[0] = 32;
                *v50 = si128;
                *(_QWORD *)(v49 + 32) += 17LL;
              }
              v27 = v69;
              v28 = v70;
              sub_2E31000(v70, v69);
              if ( !v71 )
                goto LABEL_114;
              v72(v70, v51);
              v53 = *(_WORD **)(v51 + 32);
              if ( *(_QWORD *)(v51 + 24) - (_QWORD)v53 <= 1u )
              {
                v51 = sub_CB6200(v51, (unsigned __int8 *)": ", 2u);
              }
              else
              {
                *v53 = 8250;
                *(_QWORD *)(v51 + 32) += 2LL;
              }
              v54 = (_QWORD *)v51;
              v55 = (_QWORD *)v9;
              sub_2E91850(v9, v51, 1u, 0, 0, 1, 0);
              if ( v71 )
              {
                v54 = v70;
                v55 = v70;
                v71(v70, v70, 3, v56, v57, v63);
              }
              v58 = sub_C5F790((__int64)v55, (__int64)v54);
              v59 = *(__m128i **)(v58 + 32);
              if ( *(_QWORD *)(v58 + 24) - (_QWORD)v59 <= 0x20u )
              {
                v58 = sub_CB6200(v58, "  missing input from predecessor ", 0x21u);
              }
              else
              {
                v60 = _mm_load_si128((const __m128i *)&xmmword_42F1A80);
                v59[2].m128i_i8[0] = 32;
                *v59 = v60;
                v59[1] = _mm_load_si128((const __m128i *)&xmmword_42F1A90);
                *(_QWORD *)(v58 + 32) += 33LL;
              }
              v27 = v15;
              v28 = v73;
              sub_2E31000(v73, v15);
              if ( !v74 )
                goto LABEL_114;
              v75(v73, v58);
              v61 = *(_BYTE **)(v58 + 32);
              if ( (unsigned __int64)v61 >= *(_QWORD *)(v58 + 24) )
              {
                sub_CB5D20(v58, 10);
              }
              else
              {
                *(_QWORD *)(v58 + 32) = v61 + 1;
                *v61 = 10;
              }
              if ( v74 )
                v74(v73, v73, 3);
LABEL_119:
              BUG();
            }
            v16 = *(_QWORD *)(v9 + 32);
            v17 = 1;
            while ( 1 )
            {
              v7 = 5LL * (unsigned int)(v17 + 1);
              if ( v15 == *(_QWORD *)(v16 + 40LL * (unsigned int)(v17 + 1) + 24) )
                break;
              v17 += 2;
              if ( v17 == v12 )
                goto LABEL_93;
            }
            ++v14;
          }
          while ( v13 != v14 );
LABEL_19:
          v18 = 1;
          v67 = (v11 * 8) >> 3;
          v19 = (v11 * 8) >> 5;
          v20 = &v8[v11];
          v7 = (__int64)&v8[4 * ((v11 * 8) >> 5)];
          while ( 2 )
          {
            v21 = 5LL * (unsigned int)(v18 + 1);
            v22 = *(_QWORD *)(v16 + 40LL * (unsigned int)(v18 + 1) + 24);
            if ( !a2 )
              goto LABEL_30;
            v21 = (unsigned int)v90;
            if ( !(_DWORD)v90 )
            {
              if ( v19 )
              {
                v21 = (__int64)v8;
                while ( 1 )
                {
                  if ( v22 == *(_QWORD *)v21 )
                    goto LABEL_29;
                  if ( v22 == *(_QWORD *)(v21 + 8) )
                  {
                    v21 += 8;
                    goto LABEL_29;
                  }
                  if ( v22 == *(_QWORD *)(v21 + 16) )
                  {
                    v21 += 16;
                    goto LABEL_29;
                  }
                  if ( v22 == *(_QWORD *)(v21 + 24) )
                    break;
                  v21 += 32;
                  if ( v7 == v21 )
                  {
                    v39 = ((__int64)v13 - v7) >> 3;
                    goto LABEL_60;
                  }
                }
                v21 += 24;
                goto LABEL_29;
              }
              v39 = v67;
              v21 = (__int64)v8;
LABEL_60:
              switch ( v39 )
              {
                case 2LL:
LABEL_64:
                  if ( v22 == *(_QWORD *)v21 )
                    goto LABEL_29;
                  v21 += 8;
                  break;
                case 3LL:
                  if ( v22 != *(_QWORD *)v21 )
                  {
                    v21 += 8;
                    goto LABEL_64;
                  }
LABEL_29:
                  if ( v20 == (__int64 *)v21 )
                    goto LABEL_42;
LABEL_30:
                  if ( *(int *)(v22 + 24) < 0 )
                    goto LABEL_75;
LABEL_31:
                  v18 += 2;
                  if ( v18 == v12 )
                    goto LABEL_32;
                  continue;
                case 1LL:
                  break;
                default:
LABEL_42:
                  v24 = sub_C5F790(v21, v7);
                  v25 = *(__m128i **)(v24 + 32);
                  if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0x19u )
                  {
                    v24 = sub_CB6200(v24, "Warning: malformed PHI in ", 0x1Au);
                  }
                  else
                  {
                    v26 = _mm_load_si128((const __m128i *)&xmmword_42F1AA0);
                    qmemcpy(&v25[1], "ed PHI in ", 10);
                    *v25 = v26;
                    *(_QWORD *)(v24 + 32) += 26LL;
                  }
                  v27 = v69;
                  v28 = v76;
                  sub_2E31000(v76, v69);
                  if ( !v77 )
                    goto LABEL_114;
                  v78(v76, v24);
                  v30 = *(_WORD **)(v24 + 32);
                  if ( *(_QWORD *)(v24 + 24) - (_QWORD)v30 <= 1u )
                  {
                    v24 = sub_CB6200(v24, (unsigned __int8 *)": ", 2u);
                  }
                  else
                  {
                    *v30 = 8250;
                    *(_QWORD *)(v24 + 32) += 2LL;
                  }
                  v31 = (_QWORD *)v24;
                  v32 = (_QWORD *)v9;
                  sub_2E91850(v9, v24, 1u, 0, 0, 1, 0);
                  if ( v77 )
                  {
                    v31 = v76;
                    v32 = v76;
                    v77(v76, v76, 3, v33, v34, v62);
                  }
                  v35 = sub_C5F790((__int64)v32, (__int64)v31);
                  v36 = *(__m128i **)(v35 + 32);
                  if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 0x1Eu )
                  {
                    v35 = sub_CB6200(v35, "  extra input from predecessor ", 0x1Fu);
                  }
                  else
                  {
                    v37 = _mm_load_si128((const __m128i *)&xmmword_42F1AB0);
                    qmemcpy(&v36[1], "om predecessor ", 15);
                    *v36 = v37;
                    *(_QWORD *)(v35 + 32) += 31LL;
                  }
                  v27 = v22;
                  v28 = v79;
                  sub_2E31000(v79, v22);
                  if ( !v80 )
                    goto LABEL_114;
                  v81(v79, v35);
                  v38 = *(_BYTE **)(v35 + 32);
                  if ( (unsigned __int64)v38 >= *(_QWORD *)(v35 + 24) )
                  {
                    sub_CB5D20(v35, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v35 + 32) = v38 + 1;
                    *v38 = 10;
                  }
                  if ( v80 )
                    v80(v79, v79, 3);
                  goto LABEL_119;
              }
              if ( v22 != *(_QWORD *)v21 )
                goto LABEL_42;
              goto LABEL_29;
            }
            break;
          }
          if ( !(_DWORD)v91 )
            goto LABEL_42;
          v21 = ((_DWORD)v91 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v23 = *(_QWORD *)(v89 + 8 * v21);
          if ( v22 == v23 )
            goto LABEL_30;
          for ( i = 1; ; ++i )
          {
            if ( v23 == -4096 )
              goto LABEL_42;
            v21 = ((_DWORD)v91 - 1) & (unsigned int)(i + v21);
            v23 = *(_QWORD *)(v89 + 8LL * (unsigned int)v21);
            if ( v22 == v23 )
              break;
          }
          if ( *(int *)(v22 + 24) < 0 )
          {
LABEL_75:
            v40 = sub_C5F790(v21, v7);
            v41 = *(__m128i **)(v40 + 32);
            if ( *(_QWORD *)(v40 + 24) - (_QWORD)v41 <= 0x10u )
            {
              v40 = sub_CB6200(v40, "Malformed PHI in ", 0x11u);
            }
            else
            {
              v42 = _mm_load_si128((const __m128i *)&xmmword_42F1A70);
              v41[1].m128i_i8[0] = 32;
              *v41 = v42;
              *(_QWORD *)(v40 + 32) += 17LL;
            }
            v27 = v69;
            v28 = v82;
            sub_2E31000(v82, v69);
            if ( v83 )
            {
              v84(v82, v40);
              v43 = *(_WORD **)(v40 + 32);
              if ( *(_QWORD *)(v40 + 24) - (_QWORD)v43 <= 1u )
              {
                v40 = sub_CB6200(v40, (unsigned __int8 *)": ", 2u);
              }
              else
              {
                *v43 = 8250;
                *(_QWORD *)(v40 + 32) += 2LL;
              }
              v44 = (_QWORD *)v9;
              sub_2E91850(v9, v40, 1u, 0, 0, 1, 0);
              v45 = v64;
              if ( v83 )
              {
                v45 = v82;
                v44 = v82;
                v83(v82, v82, 3);
              }
              v46 = sub_C5F790((__int64)v44, (__int64)v45);
              v47 = *(void **)(v46 + 32);
              if ( *(_QWORD *)(v46 + 24) - (_QWORD)v47 <= 0xEu )
              {
                v46 = sub_CB6200(v46, "  non-existing ", 0xFu);
              }
              else
              {
                qmemcpy(v47, "  non-existing ", 15);
                *(_QWORD *)(v46 + 32) += 15LL;
              }
              v27 = v22;
              v28 = v85;
              sub_2E31000(v85, v22);
              if ( v86 )
              {
                v87(v85, v46);
                v48 = *(_BYTE **)(v46 + 32);
                if ( (unsigned __int64)v48 >= *(_QWORD *)(v46 + 24) )
                {
                  sub_CB5D20(v46, 10);
                }
                else
                {
                  *(_QWORD *)(v46 + 32) = v48 + 1;
                  *v48 = 10;
                }
                if ( v86 )
                  v86(v85, v85, 3);
                goto LABEL_119;
              }
            }
LABEL_114:
            sub_4263D6(v28, v27, v29);
          }
          goto LABEL_31;
        }
LABEL_7:
        if ( v8 != (__int64 *)v94 )
          _libc_free((unsigned __int64)v8);
        v10 = v89;
        v7 = 8LL * (unsigned int)v91;
      }
      sub_C7D6A0(v10, v7, 8);
      result = *(_QWORD *)(v69 + 8);
      v69 = result;
    }
    while ( result != v65 );
  }
  return result;
}
