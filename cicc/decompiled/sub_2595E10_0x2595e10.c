// Function: sub_2595E10
// Address: 0x2595e10
//
__int64 __fastcall sub_2595E10(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // rax
  char (__fastcall *v6)(__int64, unsigned __int8 (__fastcall *)(__int64, _QWORD), __int64); // rcx
  _QWORD *v7; // rdi
  _QWORD *v8; // rbx
  unsigned __int8 *v9; // r12
  _BYTE *v10; // r14
  unsigned __int64 v11; // rdi
  __m128i v12; // rax
  __int64 v13; // rcx
  __int64 *v14; // r14
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // rbx
  int *v18; // r15
  _BYTE *v19; // rdi
  __int64 **v20; // r15
  _BYTE *v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rdx
  bool v25; // zf
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 *v29; // r12
  __int64 v30; // rax
  __int64 *v31; // r14
  __int64 **v32; // r11
  unsigned int v33; // esi
  __int64 v34; // r9
  __int64 v35; // r8
  unsigned int v36; // edx
  _QWORD *v37; // rdi
  _BYTE *v38; // rcx
  _BYTE *v39; // rax
  __int64 v40; // rbx
  int *v41; // r13
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rbx
  int *v45; // r13
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // r12
  _QWORD *v50; // r10
  int v51; // edi
  int v52; // ecx
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rbx
  int *v58; // r15
  unsigned int v59; // esi
  __int64 v60; // rdi
  __int64 v61; // r9
  int v62; // r8d
  __int64 v63; // r10
  _QWORD *v64; // r11
  unsigned int v65; // eax
  _QWORD *v66; // rdx
  _BYTE *v67; // rcx
  int v68; // eax
  int v69; // eax
  __int64 v70; // r8
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // [rsp-10h] [rbp-170h]
  __int64 v74; // [rsp-8h] [rbp-168h]
  int v75; // [rsp+18h] [rbp-148h]
  __int64 **v76; // [rsp+18h] [rbp-148h]
  __int64 **v77; // [rsp+18h] [rbp-148h]
  __int64 v78; // [rsp+40h] [rbp-120h]
  __int64 *v80; // [rsp+50h] [rbp-110h]
  __int64 v81; // [rsp+50h] [rbp-110h]
  char v83; // [rsp+6Bh] [rbp-F5h] BYREF
  int v84; // [rsp+6Ch] [rbp-F4h] BYREF
  unsigned __int64 v85; // [rsp+70h] [rbp-F0h] BYREF
  _BYTE *v86; // [rsp+78h] [rbp-E8h] BYREF
  int *v87; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v88; // [rsp+88h] [rbp-D8h]
  __m128i v89[2]; // [rsp+90h] [rbp-D0h] BYREF
  char v90; // [rsp+B0h] [rbp-B0h]
  __m128i v91; // [rsp+C0h] [rbp-A0h] BYREF
  _BYTE v92[32]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v93; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v94; // [rsp+F8h] [rbp-68h]
  _BYTE v95[96]; // [rsp+100h] [rbp-60h] BYREF

  v87 = &v84;
  v88 = a1;
  v94 = 0x300000000LL;
  v84 = 1;
  v93 = (__int64 *)v95;
  v85 = sub_2509740((_QWORD *)(a1 + 72));
  v2 = *(_BYTE **)(v85 - 32);
  if ( *v2 == 25 )
  {
    if ( v2[96] )
    {
      sub_2553790((const char **)v89, "ompx_no_call_asm");
      v56 = sub_B491C0(v85);
      if ( !(unsigned __int8)sub_31402A0(v56, v89) )
      {
        sub_2553790((const char **)&v91, "ompx_no_call_asm");
        if ( !(unsigned __int8)sub_31402F0(v85, &v91) )
        {
          v3 = 0;
          if ( *(_BYTE *)(a1 + 168) )
            v3 = v84;
          *(_BYTE *)(a1 + 168) = 1;
          goto LABEL_4;
        }
      }
    }
LABEL_3:
    v3 = v84;
    goto LABEL_4;
  }
  if ( sub_B491E0(v85) )
  {
    v5 = sub_2595A50(a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), a1, 1, 0, 1);
    if ( v5 )
    {
      v6 = *(char (__fastcall **)(__int64, unsigned __int8 (__fastcall *)(__int64, _QWORD), __int64))(*(_QWORD *)v5 + 112LL);
      v91.m128i_i64[0] = (__int64)&v87;
      v91.m128i_i64[1] = (__int64)&v85;
      if ( v6 == sub_2538FF0 )
      {
        if ( *(_BYTE *)(v5 + 97) )
        {
          if ( *(_BYTE *)(v5 + 296) )
          {
            v7 = *(_QWORD **)(v5 + 248);
            v8 = &v7[*(unsigned int *)(v5 + 256)];
            if ( v8 == sub_2537BA0(
                         v7,
                         (__int64)v8,
                         (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2571B50,
                         (__int64)&v91) )
              goto LABEL_3;
          }
        }
      }
      else if ( v6(v5, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2571B50, (__int64)&v91) )
      {
        goto LABEL_3;
      }
    }
  }
  v9 = (unsigned __int8 *)v85;
  v10 = *(_BYTE **)(v85 - 32);
  if ( *v10 <= 0x15u )
  {
    v57 = v88;
    v58 = v87;
    if ( *v10 )
    {
      if ( !*(_BYTE *)(v88 + 168) )
        *v87 = 0;
      if ( !*(_BYTE *)(v57 + 169) )
        *v58 = 0;
      v9 = (unsigned __int8 *)v85;
      *(_WORD *)(v57 + 168) = 257;
      goto LABEL_27;
    }
    v89[0].m128i_i64[0] = *(_QWORD *)(v85 - 32);
    v59 = *(_DWORD *)(v88 + 144);
    v60 = v88 + 120;
    if ( v59 )
    {
      v61 = v59 - 1;
      v62 = 1;
      v63 = *(_QWORD *)(v88 + 128);
      v64 = 0;
      v65 = v61 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v66 = (_QWORD *)(v63 + 8LL * v65);
      v67 = (_BYTE *)*v66;
      if ( v10 == (_BYTE *)*v66 )
        goto LABEL_27;
      while ( v67 != (_BYTE *)-4096LL )
      {
        if ( v67 == (_BYTE *)-8192LL && !v64 )
          v64 = v66;
        v65 = v61 & (v62 + v65);
        v66 = (_QWORD *)(v63 + 8LL * v65);
        v67 = (_BYTE *)*v66;
        if ( v10 == (_BYTE *)*v66 )
          goto LABEL_27;
        ++v62;
      }
      if ( v64 )
        v66 = v64;
      v91.m128i_i64[0] = (__int64)v66;
      v68 = *(_DWORD *)(v88 + 136);
      ++*(_QWORD *)(v88 + 120);
      v69 = v68 + 1;
      if ( 4 * v69 < 3 * v59 )
      {
        v70 = v59 >> 3;
        if ( v59 - *(_DWORD *)(v57 + 140) - v69 > (unsigned int)v70 )
        {
LABEL_93:
          *(_DWORD *)(v57 + 136) = v69;
          if ( *v66 != -4096 )
            --*(_DWORD *)(v57 + 140);
          *v66 = v10;
          v71 = *(unsigned int *)(v57 + 160);
          v72 = v89[0].m128i_i64[0];
          if ( v71 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 164) )
          {
            sub_C8D5F0(v57 + 152, (const void *)(v57 + 168), v71 + 1, 8u, v70, v61);
            v71 = *(unsigned int *)(v57 + 160);
          }
          *(_QWORD *)(*(_QWORD *)(v57 + 152) + 8 * v71) = v72;
          ++*(_DWORD *)(v57 + 160);
          *v58 = 0;
          v9 = (unsigned __int8 *)v85;
          goto LABEL_27;
        }
LABEL_112:
        sub_A35F10(v60, v59);
        sub_A2AFD0(v57 + 120, v89[0].m128i_i64, &v91);
        v10 = (_BYTE *)v89[0].m128i_i64[0];
        v66 = (_QWORD *)v91.m128i_i64[0];
        v69 = *(_DWORD *)(v57 + 136) + 1;
        goto LABEL_93;
      }
    }
    else
    {
      v91.m128i_i64[0] = 0;
      ++*(_QWORD *)(v88 + 120);
    }
    v59 *= 2;
    goto LABEL_112;
  }
  v11 = *(_QWORD *)(v85 - 32);
  LOBYTE(v86) = 0;
  LODWORD(v94) = 0;
  v12.m128i_i64[0] = sub_250D2C0(v11, 0);
  v91 = v12;
  if ( !(unsigned __int8)sub_2526B50(a2, &v91, a1, (__int64)&v93, 3u, &v86, 1u) )
    sub_25592F0((__int64)&v93, (__int64)v10, (__int64)v9, v13, v73, v74);
  v14 = v93;
  v15 = 2LL * (unsigned int)v94;
  v80 = &v93[v15];
  if ( v93 != &v93[v15] )
  {
    do
    {
      while ( 1 )
      {
        v17 = v88;
        v18 = v87;
        if ( !*(_BYTE *)*v14 )
          break;
        if ( !*(_BYTE *)(v88 + 168) )
          *v87 = 0;
        if ( !*(_BYTE *)(v17 + 169) )
          *v18 = 0;
        v14 += 2;
        *(_WORD *)(v17 + 168) = 257;
        if ( v80 == v14 )
          goto LABEL_26;
      }
      v89[0].m128i_i64[0] = *v14;
      sub_2571760((__int64)&v91, v88 + 120, v89[0].m128i_i64);
      if ( v92[16] )
      {
        v42 = *(unsigned int *)(v17 + 160);
        v43 = v89[0].m128i_i64[0];
        if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 164) )
        {
          v78 = v89[0].m128i_i64[0];
          sub_C8D5F0(v17 + 152, (const void *)(v17 + 168), v42 + 1, 8u, v16, v89[0].m128i_i64[0]);
          v42 = *(unsigned int *)(v17 + 160);
          v43 = v78;
        }
        *(_QWORD *)(*(_QWORD *)(v17 + 152) + 8 * v42) = v43;
        ++*(_DWORD *)(v17 + 160);
        *v18 = 0;
      }
      v14 += 2;
    }
    while ( v80 != v14 );
  }
LABEL_26:
  v9 = (unsigned __int8 *)v85;
LABEL_27:
  v91.m128i_i64[0] = (__int64)v92;
  v91.m128i_i64[1] = 0x400000000LL;
  sub_E33A00(v9, (__int64)&v91);
  v19 = (_BYTE *)v91.m128i_i64[0];
  v81 = v91.m128i_i64[0] + 8LL * v91.m128i_u32[2];
  if ( v81 != v91.m128i_i64[0] )
  {
    v20 = (__int64 **)v91.m128i_i64[0];
    while ( 1 )
    {
      v21 = (_BYTE *)**v20;
      if ( *v21 <= 0x15u )
      {
        v44 = v88;
        v45 = v87;
        if ( *v21 )
        {
          if ( !*(_BYTE *)(v88 + 168) )
            *v87 = 0;
          if ( !*(_BYTE *)(v44 + 169) )
            *v45 = 0;
          ++v20;
          *(_WORD *)(v44 + 168) = 257;
          if ( (__int64 **)v81 == v20 )
          {
LABEL_44:
            v19 = (_BYTE *)v91.m128i_i64[0];
            break;
          }
        }
        else
        {
          v86 = (_BYTE *)**v20;
          sub_2571760((__int64)v89, v88 + 120, (__int64 *)&v86);
          if ( !v90 )
            goto LABEL_43;
          v48 = *(unsigned int *)(v44 + 160);
          v49 = (__int64)v86;
          if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v44 + 164) )
          {
            sub_C8D5F0(v44 + 152, (const void *)(v44 + 168), v48 + 1, 8u, v46, v47);
            v48 = *(unsigned int *)(v44 + 160);
          }
          ++v20;
          *(_QWORD *)(*(_QWORD *)(v44 + 152) + 8 * v48) = v49;
          ++*(_DWORD *)(v44 + 160);
          *v45 = 0;
          if ( (__int64 **)v81 == v20 )
            goto LABEL_44;
        }
      }
      else
      {
        v22 = **v20;
        v83 = 0;
        v23 = v85;
        LODWORD(v94) = 0;
        v89[0].m128i_i64[0] = sub_250D2C0(v22, 0);
        v89[0].m128i_i64[1] = v24;
        v25 = (unsigned __int8)sub_2526B50(a2, v89, a1, (__int64)&v93, 3u, &v83, 1u) == 0;
        v28 = (unsigned int)v94;
        if ( v25 )
        {
          if ( (unsigned __int64)(unsigned int)v94 + 1 > HIDWORD(v94) )
          {
            sub_C8D5F0((__int64)&v93, v95, (unsigned int)v94 + 1LL, 0x10u, v26, v27);
            v28 = (unsigned int)v94;
          }
          v55 = &v93[2 * v28];
          *v55 = (__int64)v21;
          v55[1] = v23;
          v28 = (unsigned int)(v94 + 1);
          LODWORD(v94) = v94 + 1;
        }
        v29 = v93;
        v30 = 2 * v28;
        if ( v93 != &v93[v30] )
        {
          v31 = &v93[v30];
          v32 = v20;
          while ( 1 )
          {
            while ( 1 )
            {
              v39 = (_BYTE *)*v29;
              v40 = v88;
              v41 = v87;
              if ( !*(_BYTE *)*v29 )
                break;
              if ( !*(_BYTE *)(v88 + 168) )
                *v87 = 0;
              if ( !*(_BYTE *)(v40 + 169) )
                *v41 = 0;
              v29 += 2;
              *(_WORD *)(v40 + 168) = 257;
              if ( v31 == v29 )
              {
LABEL_42:
                v20 = v32;
                goto LABEL_43;
              }
            }
            v86 = (_BYTE *)*v29;
            v33 = *(_DWORD *)(v88 + 144);
            if ( !v33 )
              break;
            v34 = v33 - 1;
            v35 = *(_QWORD *)(v88 + 128);
            v36 = v34 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
            v37 = (_QWORD *)(v35 + 8LL * v36);
            v38 = (_BYTE *)*v37;
            if ( (_BYTE *)*v37 != v39 )
            {
              v75 = 1;
              v50 = 0;
              while ( v38 != (_BYTE *)-4096LL )
              {
                if ( v50 || v38 != (_BYTE *)-8192LL )
                  v37 = v50;
                v36 = v34 & (v75 + v36);
                v38 = *(_BYTE **)(v35 + 8LL * v36);
                if ( v39 == v38 )
                  goto LABEL_35;
                ++v75;
                v50 = v37;
                v37 = (_QWORD *)(v35 + 8LL * v36);
              }
              if ( !v50 )
                v50 = v37;
              v89[0].m128i_i64[0] = (__int64)v50;
              v51 = *(_DWORD *)(v88 + 136);
              ++*(_QWORD *)(v88 + 120);
              v52 = v51 + 1;
              if ( 4 * (v51 + 1) < 3 * v33 )
              {
                if ( v33 - *(_DWORD *)(v40 + 140) - v52 > v33 >> 3 )
                {
LABEL_62:
                  *(_DWORD *)(v40 + 136) = v52;
                  if ( *v50 != -4096 )
                    --*(_DWORD *)(v40 + 140);
                  *v50 = v39;
                  v53 = *(unsigned int *)(v40 + 160);
                  v54 = (__int64)v86;
                  if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 164) )
                  {
                    v77 = v32;
                    sub_C8D5F0(v40 + 152, (const void *)(v40 + 168), v53 + 1, 8u, v35, v34);
                    v53 = *(unsigned int *)(v40 + 160);
                    v32 = v77;
                  }
                  *(_QWORD *)(*(_QWORD *)(v40 + 152) + 8 * v53) = v54;
                  ++*(_DWORD *)(v40 + 160);
                  *v41 = 0;
                  goto LABEL_35;
                }
                v76 = v32;
LABEL_72:
                sub_A35F10(v40 + 120, v33);
                sub_A2AFD0(v40 + 120, (__int64 *)&v86, v89);
                v39 = v86;
                v50 = (_QWORD *)v89[0].m128i_i64[0];
                v32 = v76;
                v52 = *(_DWORD *)(v40 + 136) + 1;
                goto LABEL_62;
              }
LABEL_71:
              v76 = v32;
              v33 *= 2;
              goto LABEL_72;
            }
LABEL_35:
            v29 += 2;
            if ( v31 == v29 )
              goto LABEL_42;
          }
          v89[0].m128i_i64[0] = 0;
          ++*(_QWORD *)(v88 + 120);
          goto LABEL_71;
        }
LABEL_43:
        if ( (__int64 **)v81 == ++v20 )
          goto LABEL_44;
      }
    }
  }
  v3 = v84;
  if ( v19 != v92 )
    _libc_free((unsigned __int64)v19);
LABEL_4:
  if ( v93 != (__int64 *)v95 )
    _libc_free((unsigned __int64)v93);
  return v3;
}
