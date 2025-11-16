// Function: sub_2904650
// Address: 0x2904650
//
void __fastcall sub_2904650(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int8 **i; // r12
  unsigned __int8 *v8; // rsi
  unsigned __int8 *v9; // rax
  unsigned __int8 **v10; // rax
  __int64 v11; // r8
  __int64 v12; // rbx
  int v13; // r12d
  unsigned __int8 **v14; // r15
  __int64 v15; // rax
  bool v16; // of
  unsigned __int8 *v17; // r13
  int v18; // edx
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rax
  int v22; // eax
  __int32 v23; // r13d
  __int64 v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rax
  __m128i v27; // xmm0
  unsigned int v28; // esi
  int v29; // r13d
  __int64 v30; // rdi
  _QWORD *v31; // rax
  _BYTE *v32; // r13
  __int64 *v33; // rax
  __int64 v34; // rbx
  unsigned int v35; // esi
  unsigned int v36; // edi
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // r12
  bool v40; // zf
  _QWORD *v41; // rax
  unsigned int v42; // edx
  int v43; // ecx
  __int64 v44; // rdx
  int v45; // eax
  __int64 v46; // r13
  __int64 v47; // rbx
  int v48; // esi
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // r10
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rsi
  int v56; // eax
  __int64 *v57; // rdi
  __int64 v58; // r13
  __int64 v59; // rdx
  unsigned __int64 v60; // r14
  __int64 v61; // r13
  __int64 v62; // rbx
  __int64 *v63; // r14
  char *v64; // rax
  __int64 v65; // r13
  unsigned __int64 v66; // rdi
  int v67; // r13d
  unsigned __int8 **v68; // [rsp+30h] [rbp-1A0h]
  unsigned __int8 **v69; // [rsp+30h] [rbp-1A0h]
  unsigned __int8 **v73; // [rsp+50h] [rbp-180h]
  int v74; // [rsp+50h] [rbp-180h]
  __int64 v75; // [rsp+58h] [rbp-178h]
  __int64 v76; // [rsp+60h] [rbp-170h]
  __int64 v77; // [rsp+60h] [rbp-170h]
  unsigned __int8 *v78; // [rsp+68h] [rbp-168h]
  __int64 v79; // [rsp+68h] [rbp-168h]
  unsigned __int8 *v80; // [rsp+80h] [rbp-150h] BYREF
  __int64 v81; // [rsp+88h] [rbp-148h] BYREF
  unsigned __int64 v82; // [rsp+90h] [rbp-140h] BYREF
  int v83; // [rsp+98h] [rbp-138h]
  unsigned __int8 **v84; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-128h]
  _BYTE v86[32]; // [rsp+B0h] [rbp-120h] BYREF
  _BYTE *v87; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v88; // [rsp+D8h] [rbp-F8h]
  _BYTE v89[24]; // [rsp+E0h] [rbp-F0h] BYREF
  unsigned __int8 *v90; // [rsp+F8h] [rbp-D8h]
  __m128i v91; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v92; // [rsp+110h] [rbp-C0h] BYREF
  unsigned __int64 v93; // [rsp+118h] [rbp-B8h] BYREF
  __int64 v94; // [rsp+120h] [rbp-B0h] BYREF
  _DWORD v95[6]; // [rsp+128h] [rbp-A8h] BYREF
  unsigned __int8 *v96; // [rsp+140h] [rbp-90h]
  __m128i v97[5]; // [rsp+148h] [rbp-88h] BYREF
  char v98; // [rsp+1A0h] [rbp-30h] BYREF

  v6 = *(_QWORD *)(a1 + 32);
  v75 = v6 + 16LL * *(unsigned int *)(a1 + 40);
  if ( v6 != v75 )
  {
    for ( i = *(unsigned __int8 ***)(a1 + 32); (unsigned __int8 **)v75 != i; i += 2 )
    {
      v8 = *i;
      v9 = i[1];
      v80 = v8;
      if ( v8 != v9 )
      {
        v84 = (unsigned __int8 **)v86;
        v85 = 0x300000000LL;
        v78 = sub_28FF860((__int64)&v84, v8, v6, a4, a5, a6);
        if ( (unsigned __int64)(unsigned int)v85 - 1 > 9 )
          goto LABEL_3;
        v10 = (unsigned __int8 **)sub_1152A40(a1, (__int64 *)&v80, v6, a4, a5, a6);
        a4 = (__int64)v78;
        if ( *v10 == v78 )
          goto LABEL_14;
        if ( *v78 != 84 )
        {
          sub_1152A40(a1, (__int64 *)&v80, v6, (__int64)v78, v11, a6);
          goto LABEL_3;
        }
        v31 = (_QWORD *)sub_1152A40(a1, (__int64 *)&v80, v6, (__int64)v78, v11, a6);
        v32 = (_BYTE *)*v31;
        if ( *(_BYTE *)*v31 == 84 )
        {
          v6 = *((_DWORD *)v78 + 1) & 0x7FFFFFF;
          if ( (_DWORD)v6 == (*((_DWORD *)v32 + 1) & 0x7FFFFFF) && *((_QWORD *)v78 + 5) == *((_QWORD *)v32 + 5) )
          {
            v92 = 0;
            v33 = &v94;
            v93 = 1;
            do
            {
              *v33 = -4096;
              v33 += 2;
            }
            while ( v33 != (__int64 *)&v98 );
            if ( (_DWORD)v6 )
            {
              v34 = 0;
              v77 = 8LL * (unsigned int)v6;
              v69 = i;
              while ( 1 )
              {
                v38 = *((_QWORD *)v78 - 1);
                v39 = *(_QWORD *)(v38 + 32LL * *((unsigned int *)v78 + 18) + v34);
                v81 = *(_QWORD *)(v38 + 4 * v34);
                v40 = (unsigned __int8)sub_29013F0((__int64)&v92, &v81, &v82) == 0;
                v41 = (_QWORD *)v82;
                if ( v40 )
                  break;
LABEL_51:
                v34 += 8;
                v41[1] = v39;
                if ( v77 == v34 )
                {
                  a4 = 0;
                  i = v69;
                  v47 = *((_QWORD *)v32 - 1);
                  a6 = v93 & 1;
                  while ( 1 )
                  {
                    v53 = *(_QWORD *)(v47 + 4 * a4);
                    if ( (_BYTE)a6 )
                    {
                      v48 = 7;
                      a5 = (__int64)&v94;
                    }
                    else
                    {
                      v54 = v95[0];
                      a5 = v94;
                      if ( !v95[0] )
                        goto LABEL_84;
                      v48 = v95[0] - 1;
                    }
                    v49 = v48 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                    v50 = (__int64 *)(a5 + 16LL * v49);
                    v51 = *v50;
                    if ( v53 == *v50 )
                      goto LABEL_75;
                    v56 = 1;
                    while ( v51 != -4096 )
                    {
                      v49 = v48 & (v56 + v49);
                      v74 = v56 + 1;
                      v50 = (__int64 *)(a5 + 16LL * v49);
                      v51 = *v50;
                      if ( v53 == *v50 )
                        goto LABEL_75;
                      v56 = v74;
                    }
                    if ( (_BYTE)a6 )
                    {
                      v55 = 128;
                      goto LABEL_85;
                    }
                    v54 = v95[0];
LABEL_84:
                    v55 = 16 * v54;
LABEL_85:
                    v50 = (__int64 *)(a5 + v55);
LABEL_75:
                    v52 = 128;
                    if ( !(_BYTE)a6 )
                      v52 = 16LL * v95[0];
                    v6 = a5 + v52;
                    if ( v50 == (__int64 *)v6
                      || (v6 = v47 + 32LL * *((unsigned int *)v32 + 18), v50[1] != *(_QWORD *)(v6 + a4)) )
                    {
                      if ( !(_BYTE)a6 )
                        sub_C7D6A0(v94, 16LL * v95[0], 8);
                      goto LABEL_3;
                    }
                    a4 += 8;
                    if ( v77 == a4 )
                      goto LABEL_12;
                  }
                }
              }
              ++v92;
              v87 = (_BYTE *)v82;
              v42 = ((unsigned int)v93 >> 1) + 1;
              if ( (v93 & 1) != 0 )
              {
                v36 = 24;
                v35 = 8;
              }
              else
              {
                v35 = v95[0];
                v36 = 3 * v95[0];
              }
              if ( 4 * v42 >= v36 )
              {
                v35 *= 2;
              }
              else if ( v35 - (v42 + HIDWORD(v93)) > v35 >> 3 )
              {
LABEL_48:
                LODWORD(v93) = v93 & 1 | (2 * v42);
                if ( *v41 != -4096 )
                  --HIDWORD(v93);
                v37 = v81;
                v41[1] = 0;
                *v41 = v37;
                goto LABEL_51;
              }
              sub_2903A40((__int64)&v92, v35);
              sub_29013F0((__int64)&v92, &v81, &v87);
              v41 = v87;
              v42 = ((unsigned int)v93 >> 1) + 1;
              goto LABEL_48;
            }
            a6 = v93 & 1;
LABEL_12:
            if ( !(_BYTE)a6 )
              sub_C7D6A0(v94, 16LL * v95[0], 8);
LABEL_14:
            v73 = &v84[(unsigned int)v85];
            if ( v84 != v73 )
            {
              v68 = i;
              v12 = 0;
              v13 = 0;
              v14 = v84;
              while ( 1 )
              {
                while ( 1 )
                {
                  v17 = *v14;
                  v18 = **v14;
                  if ( (unsigned int)(v18 - 67) <= 0xC )
                  {
                    v76 = *(_QWORD *)(*((_QWORD *)v17 - 4) + 8LL);
                    sub_DFBCC0(*v14);
                    v15 = sub_DFD060(a3, (unsigned int)*v17 - 29, *((_QWORD *)v17 + 1), v76);
                    a4 = 1;
                    if ( (_DWORD)v6 == 1 )
                      v13 = 1;
                    v16 = __OFADD__(v15, v12);
                    v12 += v15;
                    if ( v16 )
                    {
                      v12 = 0x8000000000000000LL;
                      if ( v15 > 0 )
                        v12 = 0x7FFFFFFFFFFFFFFFLL;
                    }
                    goto LABEL_19;
                  }
                  if ( (_BYTE)v18 != 63 )
                    BUG();
                  v19 = sub_DFDB90((__int64)a3);
                  if ( v20 == 1 )
                    v13 = 1;
                  v16 = __OFADD__(v19, v12);
                  v12 += v19;
                  if ( v16 )
                    break;
LABEL_25:
                  if ( (unsigned __int8)sub_B4DD90((__int64)v17) )
                    goto LABEL_19;
                  v16 = __OFADD__(2, v12);
                  v21 = v12 + 2;
                  v12 = 0x7FFFFFFFFFFFFFFFLL;
                  if ( v16 )
                    goto LABEL_19;
                  v12 = v21;
LABEL_28:
                  if ( v73 == ++v14 )
                  {
LABEL_29:
                    v22 = v13;
                    i = v68;
                    goto LABEL_30;
                  }
                }
                if ( v19 > 0 )
                {
                  v12 = 0x7FFFFFFFFFFFFFFFLL;
                  goto LABEL_25;
                }
                v12 = 0x8000000000000000LL;
                if ( !(unsigned __int8)sub_B4DD90((__int64)v17) )
                {
                  v12 = 0x8000000000000002LL;
                  goto LABEL_28;
                }
LABEL_19:
                if ( v73 == ++v14 )
                  goto LABEL_29;
              }
            }
            v22 = 0;
            v12 = 0;
LABEL_30:
            v23 = v22;
            v87 = v89;
            v88 = 0x300000000LL;
            v91.m128i_i64[0] = 0;
            v91.m128i_i32[2] = 0;
            sub_28FED60((__int64)&v87, (__int64)&v84, v6, a4, (__int64)&v87, a6);
            v26 = (__int64)v80;
            v91.m128i_i64[0] = v12;
            v90 = v78;
            v91.m128i_i32[2] = v23;
            v92 = (__int64)v80;
            v93 = (unsigned __int64)v95;
            v94 = 0x300000000LL;
            if ( (_DWORD)v88 )
            {
              sub_28FED60((__int64)&v93, (__int64)&v87, v24, 0x300000000LL, (__int64)&v87, v25);
              v78 = v90;
              v26 = v92;
            }
            v27 = _mm_loadu_si128(&v91);
            v82 = v26;
            v83 = 0;
            v96 = v78;
            v97[0] = v27;
            v28 = *(_DWORD *)(a2 + 24);
            if ( v28 )
            {
              a6 = *(_QWORD *)(a2 + 8);
              v29 = 1;
              v30 = 0;
              a4 = (v28 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v6 = a6 + 16 * a4;
              a5 = *(_QWORD *)v6;
              if ( *(_QWORD *)v6 == v26 )
              {
LABEL_34:
                if ( (_DWORD *)v93 != v95 )
                  _libc_free(v93);
                if ( v87 != v89 )
                  _libc_free((unsigned __int64)v87);
                goto LABEL_3;
              }
              while ( a5 != -4096 )
              {
                if ( a5 != -8192 || v30 )
                  v6 = v30;
                a4 = (v28 - 1) & (v29 + (_DWORD)a4);
                a5 = *(_QWORD *)(a6 + 16LL * (unsigned int)a4);
                if ( v26 == a5 )
                  goto LABEL_34;
                ++v29;
                v30 = v6;
                v6 = a6 + 16LL * (unsigned int)a4;
              }
              if ( !v30 )
                v30 = v6;
              ++*(_QWORD *)a2;
              v43 = *(_DWORD *)(a2 + 16);
              v81 = v30;
              a5 = (unsigned int)(v43 + 1);
              if ( 4 * (int)a5 < 3 * v28 )
              {
                if ( v28 - *(_DWORD *)(a2 + 20) - (unsigned int)a5 > v28 >> 3 )
                {
LABEL_64:
                  *(_DWORD *)(a2 + 16) = a5;
                  if ( *(_QWORD *)v30 != -4096 )
                    --*(_DWORD *)(a2 + 20);
                  *(_QWORD *)v30 = v26;
                  *(_DWORD *)(v30 + 8) = v83;
                  *(_DWORD *)(v30 + 8) = *(_DWORD *)(a2 + 40);
                  v44 = *(unsigned int *)(a2 + 40);
                  v45 = v44;
                  if ( *(_DWORD *)(a2 + 44) <= (unsigned int)v44 )
                  {
                    v79 = sub_C8D7D0(a2 + 32, a2 + 48, 0, 0x48u, &v82, a6);
                    a5 = 72LL * *(unsigned int *)(a2 + 40);
                    v57 = (__int64 *)(a5 + v79);
                    v58 = a5 + v79;
                    if ( a5 + v79 )
                    {
                      v59 = v92;
                      v57[2] = 0x300000000LL;
                      *v57 = v59;
                      v57[1] = (__int64)(v57 + 3);
                      a4 = (unsigned int)v94;
                      if ( (_DWORD)v94 )
                        sub_28FEE40((__int64)(v57 + 1), (char **)&v93, (__int64)(v57 + 3), (unsigned int)v94, a5, a6);
                      *(_QWORD *)(v58 + 48) = v96;
                      *(__m128i *)(v58 + 56) = _mm_loadu_si128(v97);
                      a5 = 72LL * *(unsigned int *)(a2 + 40);
                    }
                    v6 = *(_QWORD *)(a2 + 32);
                    v60 = v6 + a5;
                    if ( v6 != v6 + a5 )
                    {
                      v61 = v79;
                      v62 = v6 + a5;
                      v63 = *(__int64 **)(a2 + 32);
                      do
                      {
                        if ( v61 )
                        {
                          v64 = (char *)*v63;
                          *(_DWORD *)(v61 + 16) = 0;
                          *(_DWORD *)(v61 + 20) = 3;
                          *(_QWORD *)v61 = v64;
                          *(_QWORD *)(v61 + 8) = v61 + 24;
                          v6 = *((unsigned int *)v63 + 4);
                          if ( (_DWORD)v6 )
                            sub_28FEE40(v61 + 8, (char **)v63 + 1, v6, a4, a5, a6);
                          *(_QWORD *)(v61 + 48) = v63[6];
                          *(__m128i *)(v61 + 56) = _mm_loadu_si128((const __m128i *)(v63 + 7));
                        }
                        v63 += 9;
                        v61 += 72;
                      }
                      while ( (__int64 *)v62 != v63 );
                      v65 = *(_QWORD *)(a2 + 32);
                      v60 = v65 + 72LL * *(unsigned int *)(a2 + 40);
                      if ( v65 != v60 )
                      {
                        do
                        {
                          v60 -= 72LL;
                          v66 = *(_QWORD *)(v60 + 8);
                          if ( v66 != v60 + 24 )
                            _libc_free(v66);
                        }
                        while ( v65 != v60 );
                        v60 = *(_QWORD *)(a2 + 32);
                      }
                    }
                    v67 = v82;
                    if ( v60 != a2 + 48 )
                      _libc_free(v60);
                    ++*(_DWORD *)(a2 + 40);
                    *(_QWORD *)(a2 + 32) = v79;
                    *(_DWORD *)(a2 + 44) = v67;
                  }
                  else
                  {
                    a4 = 9 * v44;
                    v6 = *(_QWORD *)(a2 + 32);
                    v46 = v6 + 8 * a4;
                    if ( v46 )
                    {
                      *(_QWORD *)v46 = v92;
                      *(_QWORD *)(v46 + 8) = v46 + 24;
                      *(_QWORD *)(v46 + 16) = 0x300000000LL;
                      if ( (_DWORD)v94 )
                        sub_28FEE40(v46 + 8, (char **)&v93, v6, a4, a5, a6);
                      *(_QWORD *)(v46 + 48) = v96;
                      *(__m128i *)(v46 + 56) = _mm_loadu_si128(v97);
                      v45 = *(_DWORD *)(a2 + 40);
                    }
                    *(_DWORD *)(a2 + 40) = v45 + 1;
                  }
                  goto LABEL_34;
                }
LABEL_97:
                sub_D39D40(a2, v28);
                sub_22B1A50(a2, (__int64 *)&v82, &v81);
                v26 = v82;
                a5 = (unsigned int)(*(_DWORD *)(a2 + 16) + 1);
                v30 = v81;
                goto LABEL_64;
              }
            }
            else
            {
              v81 = 0;
              ++*(_QWORD *)a2;
            }
            v28 *= 2;
            goto LABEL_97;
          }
        }
LABEL_3:
        if ( v84 != (unsigned __int8 **)v86 )
          _libc_free((unsigned __int64)v84);
      }
    }
  }
}
