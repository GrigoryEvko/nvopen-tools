// Function: sub_37E9130
// Address: 0x37e9130
//
__int64 __fastcall sub_37E9130(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r12
  __int64 (*v5)(); // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  _QWORD *v10; // rdx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  unsigned int v17; // ebx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  _DWORD *v21; // rax
  _DWORD *i; // rdx
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // r13
  int v30; // eax
  unsigned __int64 v31; // r13
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // rbx
  int v35; // eax
  char v36; // dl
  unsigned __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  int v42; // eax
  int v43; // eax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // r12
  __int64 v48; // rdi
  __int64 v49; // r13
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdi
  _QWORD *v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // r15
  __int64 v59; // rdx
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // r9
  __int64 v62; // rsi
  char *v63; // r15
  char *v64; // rdi
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rdi
  __int64 (*v73)(); // rax
  int v74; // eax
  unsigned __int8 v75; // r12
  int v76; // eax
  __int64 v77; // rax
  int *v78; // rax
  int *v79; // rbx
  __int64 v80; // rdi
  char (__fastcall *v81)(__int64, __int64); // rax
  int v82; // eax
  int v84; // eax
  char v85; // al
  unsigned __int8 j; // [rsp+27h] [rbp-79h]
  __int64 m128i_i64; // [rsp+28h] [rbp-78h]
  __int64 v88; // [rsp+30h] [rbp-70h]
  __int64 v89; // [rsp+38h] [rbp-68h]
  unsigned __int8 v90; // [rsp+40h] [rbp-60h]
  __int64 *v91; // [rsp+40h] [rbp-60h]
  int *v92; // [rsp+48h] [rbp-58h]
  __int64 v93; // [rsp+50h] [rbp-50h]
  __int64 v94; // [rsp+58h] [rbp-48h]
  __int64 v95; // [rsp+60h] [rbp-40h] BYREF
  int *v96; // [rsp+68h] [rbp-38h]

  v2 = a2;
  a1[31].m128i_i64[1] = a2;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(_QWORD, __int64, _QWORD))v5)(*(_QWORD *)(a2 + 16), a2, 0);
    v2 = a1[31].m128i_i64[1];
  }
  a1[32].m128i_i64[1] = v6;
  a1[33].m128i_i64[0] = *(_QWORD *)(v2 + 8);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 200LL))(v4);
  a1[32].m128i_i64[0] = v7;
  v8 = v7;
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 528LL);
  if ( v9 == sub_2FF52D0 || ((unsigned __int8 (__fastcall *)(__int64, __int64))v9)(v8, a1[31].m128i_i64[1]) )
  {
    v10 = (_QWORD *)sub_22077B0(0xA8u);
    if ( v10 )
    {
      memset(v10, 0, 0xA8u);
      v10[5] = v10 + 7;
      v10[6] = 0x200000000LL;
      v10[12] = v10 + 14;
      v10[13] = 0x600000000LL;
    }
    v11 = a1[27].m128i_u64[0];
    a1[27].m128i_i64[0] = (__int64)v10;
    if ( v11 )
    {
      v12 = *(_QWORD *)(v11 + 96);
      if ( v12 != v11 + 112 )
        _libc_free(v12);
      v13 = *(_QWORD *)(v11 + 40);
      if ( v13 != v11 + 56 )
        _libc_free(v13);
      j_j___libc_free_0(v11);
    }
  }
  sub_2E7A760(a1[31].m128i_i64[1], 0);
  v16 = a1[31].m128i_i64[1];
  a1[13].m128i_i32[0] = 0;
  v89 = (__int64)&a1[12].m128i_i64[1];
  v17 = (__int64)(*(_QWORD *)(v16 + 104) - *(_QWORD *)(v16 + 96)) >> 3;
  v18 = (__int64)(*(_QWORD *)(v16 + 104) - *(_QWORD *)(v16 + 96)) >> 3;
  if ( v17 )
  {
    v19 = 0;
    if ( v17 > (unsigned __int64)a1[13].m128i_u32[1] )
    {
      sub_C8D5F0(v89, &a1[13].m128i_u64[1], v17, 8u, v14, v15);
      v19 = 8LL * a1[13].m128i_u32[0];
    }
    v20 = a1[12].m128i_i64[1];
    v21 = (_DWORD *)(v20 + v19);
    for ( i = (_DWORD *)(v20 + 8LL * v17); i != v21; v21 += 2 )
    {
      if ( v21 )
      {
        *v21 = 0;
        v21[1] = 0;
      }
    }
    a1[13].m128i_i32[0] = v18;
  }
  a1[21].m128i_i64[1] = 0;
  m128i_i64 = (__int64)a1[22].m128i_i64;
  sub_37E7E80((__int64)a1[22].m128i_i64);
  v23 = a1[31].m128i_i64[1];
  v24 = *(_QWORD *)(v23 + 328);
  v25 = v23 + 320;
  if ( v24 == v25 )
  {
    v27 = v25;
  }
  else
  {
    do
    {
      while ( 1 )
      {
        *(_DWORD *)(a1[12].m128i_i64[1] + 8LL * *(int *)(v24 + 24) + 4) = sub_37E70D0((__int64)a1, v24);
        if ( *(_DWORD *)(v24 + 252) == unk_501EB38 && *(_DWORD *)(v24 + 256) == unk_501EB3C )
          break;
        a1[21].m128i_i64[1] = v24;
        v24 = *(_QWORD *)(v24 + 8);
        if ( v25 == v24 )
          goto LABEL_26;
      }
      v24 = *(_QWORD *)(v24 + 8);
    }
    while ( v25 != v24 );
LABEL_26:
    v26 = a1[31].m128i_i64[1];
    v25 = *(_QWORD *)(v26 + 328);
    v27 = v26 + 320;
  }
  sub_37E6D50((__int64)a1, v25, v27);
  for ( j = 0; ; j = v90 )
  {
    v28 = a1[31].m128i_i64[1];
    v88 = v28 + 320;
    v93 = *(_QWORD *)(v28 + 328);
    if ( v93 == v28 + 320 )
      break;
    v90 = 0;
    do
    {
      v29 = sub_2E31A10(v93, 1);
      v94 = v93 + 48;
      if ( v29 == v93 + 48 )
        goto LABEL_30;
      v30 = *(_DWORD *)(v29 + 44);
      if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL) & 0x400LL) == 0 )
          goto LABEL_35;
      }
      else if ( !sub_2E88A90(v29, 1024, 1) )
      {
        goto LABEL_35;
      }
      v74 = *(_DWORD *)(v29 + 44);
      if ( (v74 & 4) != 0 || (v74 & 8) == 0 )
        v75 = BYTE1(*(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL)) & 1;
      else
        v75 = sub_2E88A90(v29, 256, 1);
      if ( v75 )
      {
        v76 = *(_DWORD *)(v29 + 44);
        if ( (v76 & 4) != 0 || (v76 & 8) == 0 )
          v77 = (*(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL) >> 11) & 1LL;
        else
          LOBYTE(v77) = sub_2E88A90(v29, 2048, 1);
        if ( !(_BYTE)v77 )
        {
          v78 = (int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1[32].m128i_i64[1] + 328LL))(
                         a1[32].m128i_i64[1],
                         v29);
          v79 = v78;
          if ( v78 )
          {
            if ( !(unsigned __int8)sub_37E6CB0(a1, v29, v78) )
            {
              v80 = a1[32].m128i_i64[1];
              v81 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)v80 + 1328LL);
              if ( v81 == sub_2FDE950 )
              {
                v82 = *(_DWORD *)(v29 + 44);
                if ( (v82 & 4) != 0 || (v82 & 8) == 0 )
                {
                  if ( (*(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL) & 0x20LL) == 0 )
                  {
LABEL_107:
                    v96 = v79;
                    v95 = v93;
                    if ( !sub_37E7DB0(m128i_i64, &v95) )
                    {
                      sub_37E8A20(a1, v29);
                      v90 = v75;
                    }
                    goto LABEL_35;
                  }
                }
                else if ( !sub_2E88A90(v29, 32, 1) )
                {
                  goto LABEL_107;
                }
                v84 = *(_DWORD *)(v29 + 44);
                if ( (v84 & 4) != 0 || (v84 & 8) == 0 )
                  v85 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v29 + 16) + 24LL) >> 7;
                else
                  v85 = sub_2E88A90(v29, 128, 1);
              }
              else
              {
                v85 = v81(v80, v29);
              }
              if ( v85 )
                goto LABEL_35;
              goto LABEL_107;
            }
          }
        }
      }
LABEL_35:
      v31 = sub_2E313E0(v93);
      while ( v94 != v31 )
      {
        if ( !v31 )
          BUG();
        v35 = *(_DWORD *)(v31 + 44);
        v36 = v35;
        if ( (*(_BYTE *)v31 & 4) != 0 )
        {
          v32 = *(_QWORD *)(v31 + 8);
          if ( (v35 & 4) != 0 || (v35 & 8) == 0 )
            goto LABEL_39;
        }
        else
        {
          if ( (v35 & 8) == 0 )
          {
            v32 = *(_QWORD *)(v31 + 8);
LABEL_39:
            v33 = *(_QWORD *)(v31 + 16);
            v34 = (*(_QWORD *)(v33 + 24) >> 10) & 1LL;
            if ( (*(_QWORD *)(v33 + 24) & 0x400LL) == 0 )
              goto LABEL_40;
            goto LABEL_49;
          }
          v37 = v31;
          do
            v37 = *(_QWORD *)(v37 + 8);
          while ( (*(_BYTE *)(v37 + 44) & 8) != 0 );
          v32 = *(_QWORD *)(v37 + 8);
          if ( (v36 & 4) != 0 )
            goto LABEL_39;
        }
        LOBYTE(v34) = sub_2E88A90(v31, 1024, 1);
        if ( !(_BYTE)v34 )
          goto LABEL_40;
LABEL_49:
        v38 = *(_DWORD *)(v31 + 44);
        if ( (v38 & 4) != 0 || (v38 & 8) == 0 )
          v39 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 24LL) >> 8) & 1LL;
        else
          LOBYTE(v39) = sub_2E88A90(v31, 256, 1);
        if ( !(_BYTE)v39 )
        {
          v40 = *(_DWORD *)(v31 + 44);
          if ( (v40 & 4) != 0 || (v40 & 8) == 0 )
            v41 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 24LL) >> 11) & 1LL;
          else
            LOBYTE(v41) = sub_2E88A90(v31, 2048, 1);
          if ( !(_BYTE)v41 && *(_WORD *)(v31 + 68) != 34 )
          {
            v92 = (int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1[32].m128i_i64[1] + 328LL))(
                           a1[32].m128i_i64[1],
                           v31);
            if ( !(unsigned __int8)sub_37E6CB0(a1, v31, v92) )
            {
              if ( v94 == v32 )
                goto LABEL_63;
              v42 = *(_DWORD *)(v32 + 44);
              if ( (v42 & 4) != 0 || (v42 & 8) == 0 )
              {
                if ( (*(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL) & 0x400LL) != 0 )
                  goto LABEL_69;
              }
              else
              {
                if ( !sub_2E88A90(v32, 1024, 1) )
                  goto LABEL_63;
LABEL_69:
                v43 = *(_DWORD *)(v32 + 44);
                if ( (v43 & 4) != 0 || (v43 & 8) == 0 )
                  v44 = (*(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL) >> 8) & 1LL;
                else
                  LOBYTE(v44) = sub_2E88A90(v32, 256, 1);
                if ( !(_BYTE)v44 )
                {
                  v45 = *(_DWORD *)(v32 + 44);
                  if ( (v45 & 4) != 0 || (v45 & 8) == 0 )
                    v46 = (*(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL) >> 11) & 1LL;
                  else
                    LOBYTE(v46) = sub_2E88A90(v32, 2048, 1);
                  if ( !(_BYTE)v46 )
                  {
                    v47 = *(_QWORD *)(v32 + 24);
                    v48 = a1[31].m128i_i64[1];
                    LOBYTE(v96) = 0;
                    v49 = sub_2E7AAE0(v48, *(_QWORD *)(v47 + 16), v95, 0);
                    v91 = *(__int64 **)(v47 + 8);
                    sub_2E33BD0(a1[31].m128i_i64[1] + 320, v49);
                    v50 = *v91;
                    v51 = *(_QWORD *)v49 & 7LL;
                    *(_QWORD *)(v49 + 8) = v91;
                    v50 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)v49 = v50 | v51;
                    *(_QWORD *)(v50 + 8) = v49;
                    *v91 = v49 | *v91 & 7;
                    *(_QWORD *)(v49 + 252) = *(_QWORD *)(v47 + 252);
                    *(_BYTE *)(v49 + 261) = *(_BYTE *)(v47 + 261);
                    *(_BYTE *)(v47 + 261) = 0;
                    if ( v47 + 48 != v32 && v47 + 48 != v49 + 48 )
                    {
                      sub_2E310C0((__int64 *)(v49 + 40), (__int64 *)(v47 + 40), v32, v47 + 48);
                      v52 = *(_QWORD *)(v47 + 48);
                      *(_QWORD *)((*(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v47 + 48;
                      v52 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v47 + 48) = *(_QWORD *)(v47 + 48) & 7LL | *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
                      v53 = *(_QWORD *)(v49 + 48);
                      *(_QWORD *)(v52 + 8) = v49 + 48;
                      v53 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)v32 = v53 | *(_QWORD *)v32 & 7LL;
                      *(_QWORD *)(v53 + 8) = v32;
                      *(_QWORD *)(v49 + 48) = v52 | *(_QWORD *)(v49 + 48) & 7LL;
                    }
                    v54 = a1[32].m128i_i64[1];
                    v95 = 0;
                    (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v54 + 368LL))(
                      v54,
                      v47,
                      v49,
                      0,
                      0,
                      0,
                      &v95,
                      0);
                    if ( v95 )
                      sub_B91220((__int64)&v95, v95);
                    v56 = a1[13].m128i_u32[0];
                    v57 = 8 * v56;
                    v58 = 8LL * *(int *)(v49 + 24);
                    LODWORD(v59) = a1[13].m128i_i32[0];
                    v60 = a1[13].m128i_u32[1];
                    v61 = v56 + 1;
                    if ( v58 == v57 )
                    {
                      if ( v60 < v61 )
                      {
                        sub_C8D5F0(v89, &a1[13].m128i_u64[1], v61, 8u, (__int64)v55, v61);
                        v57 = 8LL * a1[13].m128i_u32[0];
                      }
                      v65 = a1[12].m128i_i64[1];
                      *(_QWORD *)(v65 + v57) = 0;
                      ++a1[13].m128i_i32[0];
                    }
                    else
                    {
                      if ( v60 < v61 )
                      {
                        sub_C8D5F0(v89, &a1[13].m128i_u64[1], v61, 8u, (__int64)v55, v61);
                        v59 = a1[13].m128i_u32[0];
                        v57 = 8 * v59;
                      }
                      v62 = a1[12].m128i_i64[1];
                      v63 = (char *)(v62 + v58);
                      v64 = (char *)(v62 + v57 - 8);
                      v55 = (_QWORD *)(v57 + v62);
                      if ( v57 + v62 )
                      {
                        *v55 = *(_QWORD *)v64;
                        v62 = a1[12].m128i_i64[1];
                        v59 = a1[13].m128i_u32[0];
                        v57 = 8 * v59;
                        v64 = (char *)(v62 + 8 * v59 - 8);
                      }
                      if ( v63 != v64 )
                      {
                        memmove((void *)(v62 + v57 - (v64 - v63)), v63, v64 - v63);
                        LODWORD(v59) = a1[13].m128i_i32[0];
                      }
                      v65 = (unsigned int)(v59 + 1);
                      a1[13].m128i_i32[0] = v65;
                      *(_QWORD *)v63 = 0;
                    }
                    sub_2E340B0(v49, v47, v65, v60, (__int64)v55, v61);
                    sub_2E33F80(v47, v49, -1, v66, v67, v68);
                    sub_2E33F80(v47, (__int64)v92, -1, v69, v70, v71);
                    sub_2E32A60(v47, v49);
                    *(_DWORD *)(a1[12].m128i_i64[1] + 8LL * *(int *)(v47 + 24) + 4) = sub_37E70D0((__int64)a1, v47);
                    *(_DWORD *)(a1[12].m128i_i64[1] + 8LL * *(int *)(v49 + 24) + 4) = sub_37E70D0((__int64)a1, v49);
                    sub_37E6D50((__int64)a1, v47, *(_QWORD *)(v49 + 8));
                    v72 = a1[32].m128i_i64[0];
                    v73 = *(__int64 (**)())(*(_QWORD *)v72 + 528LL);
                    if ( v73 == sub_2FF52D0
                      || ((unsigned __int8 (__fastcall *)(__int64, __int64))v73)(v72, a1[31].m128i_i64[1]) )
                    {
                      sub_3509790(&a1[27].m128i_i64[1], (_QWORD *)v49);
                    }
                    goto LABEL_64;
                  }
                }
              }
LABEL_63:
              sub_37E7470(a1, v31);
LABEL_64:
              v90 = v34;
              v31 = sub_2E313E0(v93);
              continue;
            }
          }
        }
LABEL_40:
        v31 = v32;
      }
LABEL_30:
      v93 = *(_QWORD *)(v93 + 8);
    }
    while ( v88 != v93 );
    if ( !v90 )
      break;
    sub_37E6D50((__int64)a1, *(_QWORD *)(a1[31].m128i_i64[1] + 328), a1[31].m128i_i64[1] + 320);
  }
  a1[13].m128i_i32[0] = 0;
  sub_37E7E80(m128i_i64);
  return j;
}
