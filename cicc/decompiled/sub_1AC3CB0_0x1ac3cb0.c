// Function: sub_1AC3CB0
// Address: 0x1ac3cb0
//
_QWORD *__fastcall sub_1AC3CB0(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        __m128 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // r12
  unsigned int v16; // r13d
  char *v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  _QWORD *v21; // rax
  __int64 v22; // r12
  unsigned int v23; // eax
  __int64 v24; // rbx
  __int64 v25; // rdi
  __int64 v26; // r15
  __int64 v27; // rcx
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rcx
  unsigned int v31; // r10d
  _QWORD *v32; // rdx
  __int64 v33; // r9
  _QWORD *v34; // rcx
  __int64 v35; // rax
  _QWORD *v36; // r13
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 i; // r12
  int v41; // ecx
  unsigned int v42; // ecx
  _QWORD *v43; // rdi
  char *v45; // rdx
  _QWORD *v46; // rax
  __int64 v47; // r13
  int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // rdx
  _QWORD *v51; // rax
  __int64 v52; // r9
  unsigned int v53; // edx
  __int64 v54; // rdi
  int v55; // r11d
  _QWORD *v56; // r9
  char *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rbx
  unsigned int v60; // r8d
  __int64 v61; // rsi
  _QWORD *v62; // r10
  int v63; // r11d
  _QWORD *v64; // r9
  unsigned int v65; // eax
  double v66; // xmm4_8
  double v67; // xmm5_8
  int v68; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v69; // [rsp+8h] [rbp-A8h]
  _QWORD *v70; // [rsp+8h] [rbp-A8h]
  __int64 v71; // [rsp+8h] [rbp-A8h]
  _QWORD *v73; // [rsp+20h] [rbp-90h]
  _QWORD *v74; // [rsp+28h] [rbp-88h]
  _QWORD *v75; // [rsp+28h] [rbp-88h]
  __int64 v76; // [rsp+28h] [rbp-88h]
  unsigned int v77; // [rsp+28h] [rbp-88h]
  _QWORD v78[2]; // [rsp+30h] [rbp-80h] BYREF
  const char *v79; // [rsp+40h] [rbp-70h] BYREF
  char *v80; // [rsp+48h] [rbp-68h]
  __int16 v81; // [rsp+50h] [rbp-60h]
  __int128 v82; // [rsp+60h] [rbp-50h] BYREF
  __int64 v83; // [rsp+70h] [rbp-40h]
  unsigned int v84; // [rsp+78h] [rbp-38h]

  v11 = a1;
  v73 = *(_QWORD **)(a1 + 8);
  if ( !v73 )
  {
    sub_15F20C0((_QWORD *)a1);
    return v73;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
  v14 = sub_1632FA0(*(_QWORD *)(v13 + 40));
  v15 = *(_QWORD **)a1;
  v16 = *(_DWORD *)(v14 + 4);
  if ( a3 )
  {
    v79 = sub_1649960(a1);
    LOWORD(v83) = 773;
    *(_QWORD *)&v82 = &v79;
    v80 = v17;
    *((_QWORD *)&v82 + 1) = ".reg2mem";
    v18 = sub_1648A60(64, 1u);
    v73 = v18;
    if ( v18 )
      sub_15F8BC0((__int64)v18, v15, v16, 0, (__int64)&v82, a3);
  }
  else
  {
    v79 = sub_1649960(a1);
    *(_QWORD *)&v82 = &v79;
    v80 = v57;
    LOWORD(v83) = 773;
    *((_QWORD *)&v82 + 1) = ".reg2mem";
    v58 = *(_QWORD *)(v13 + 80);
    if ( !v58 )
      BUG();
    v59 = *(_QWORD *)(v58 + 24);
    if ( v59 )
      v59 -= 24;
    v73 = sub_1648A60(64, 1u);
    if ( v73 )
      sub_15F8BC0((__int64)v73, v15, v16, 0, (__int64)&v82, v59);
  }
  if ( *(_BYTE *)(a1 + 16) == 29 && !sub_157F0B0(*(_QWORD *)(a1 - 48)) )
  {
    v65 = sub_137DFF0(*(_QWORD *)(a1 + 40), *(_QWORD *)(a1 - 48));
    v82 = 0u;
    LODWORD(v83) = (_DWORD)&loc_1000000;
    sub_1AAC5F0(a1, v65, &v82, a4, a5, a6, a7, v66, v67, a10, a11);
  }
  v19 = *(_QWORD *)(a1 + 8);
  v20 = v11;
  if ( v19 )
  {
    while ( 1 )
    {
      v21 = sub_1648700(v19);
      v22 = (__int64)v21;
      if ( *((_BYTE *)v21 + 16) == 77 )
      {
        v82 = 0u;
        v83 = 0;
        v84 = 0;
        v23 = *((_DWORD *)v21 + 5) & 0xFFFFFFF;
        if ( v23 )
        {
          v24 = 0;
          v25 = 0;
          v26 = 8LL * v23;
          while ( 1 )
          {
            while ( 1 )
            {
              v29 = *(_BYTE *)(v22 + 23) & 0x40;
              v27 = v29 ? *(_QWORD *)(v22 - 8) : v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
              v28 = *(_QWORD *)(v27 + 3 * v24);
              if ( v28 )
              {
                if ( v28 == v20 )
                  break;
              }
              v24 += 8;
              if ( v26 == v24 )
                goto LABEL_31;
            }
            v30 = *(_QWORD *)(v24 + v27 + 24LL * *(unsigned int *)(v22 + 56) + 8);
            if ( !v84 )
              break;
            v31 = (v84 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v32 = (_QWORD *)(v25 + 16LL * v31);
            v33 = *v32;
            if ( v30 != *v32 )
            {
              v68 = 1;
              v74 = 0;
              while ( v33 != -8 )
              {
                if ( v33 == -16 )
                {
                  if ( v74 )
                    v32 = v74;
                  v74 = v32;
                }
                v31 = (v84 - 1) & (v68 + v31);
                v32 = (_QWORD *)(v25 + 16LL * v31);
                v33 = *v32;
                if ( v30 == *v32 )
                  goto LABEL_20;
                ++v68;
              }
              if ( v74 )
                v32 = v74;
              *(_QWORD *)&v82 = v82 + 1;
              v48 = v83 + 1;
              v75 = v32;
              if ( 4 * ((int)v83 + 1) < 3 * v84 )
              {
                if ( v84 - HIDWORD(v83) - v48 <= v84 >> 3 )
                {
                  v77 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
                  v71 = v30;
                  sub_141A900((__int64)&v82, v84);
                  if ( !v84 )
                  {
LABEL_108:
                    LODWORD(v83) = v83 + 1;
                    BUG();
                  }
                  v30 = v71;
                  v60 = (v84 - 1) & v77;
                  v75 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v60);
                  v61 = *v75;
                  v48 = v83 + 1;
                  if ( v71 != *v75 )
                  {
                    v62 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v60);
                    v63 = 1;
                    v64 = 0;
                    while ( v61 != -8 )
                    {
                      if ( !v64 && v61 == -16 )
                        v64 = v62;
                      v60 = (v84 - 1) & (v63 + v60);
                      v62 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v60);
                      v61 = *v62;
                      if ( v71 == *v62 )
                      {
                        v75 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v60);
                        goto LABEL_56;
                      }
                      ++v63;
                    }
                    if ( !v64 )
                      v64 = v62;
                    v75 = v64;
                  }
                }
                goto LABEL_56;
              }
LABEL_68:
              v76 = v30;
              sub_141A900((__int64)&v82, 2 * v84);
              if ( !v84 )
                goto LABEL_108;
              v30 = v76;
              v53 = (v84 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
              v75 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v53);
              v54 = *v75;
              v48 = v83 + 1;
              if ( v30 != *v75 )
              {
                v55 = 1;
                v56 = 0;
                while ( v54 != -8 )
                {
                  if ( v54 == -16 && !v56 )
                    v56 = v75;
                  v53 = (v84 - 1) & (v55 + v53);
                  v75 = (_QWORD *)(*((_QWORD *)&v82 + 1) + 16LL * v53);
                  v54 = *v75;
                  if ( v30 == *v75 )
                    goto LABEL_56;
                  ++v55;
                }
                if ( !v56 )
                  v56 = v75;
                v75 = v56;
              }
LABEL_56:
              LODWORD(v83) = v48;
              if ( *v75 != -8 )
                --HIDWORD(v83);
              *v75 = v30;
              v75[1] = 0;
              goto LABEL_59;
            }
LABEL_20:
            v34 = (_QWORD *)v32[1];
            if ( !v34 )
            {
              v75 = v32;
LABEL_59:
              v78[0] = sub_1649960(v20);
              v78[1] = v49;
              v79 = (const char *)v78;
              v81 = 773;
              v80 = ".reload";
              if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
                v50 = *(_QWORD *)(v22 - 8);
              else
                v50 = v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
              v69 = sub_157EBA0(*(_QWORD *)(v24 + v50 + 24LL * *(unsigned int *)(v22 + 56) + 8));
              v51 = sub_1648A60(64, 1u);
              if ( v51 )
              {
                v52 = v69;
                v70 = v51;
                sub_15F90C0((__int64)v51, *(_QWORD *)(*v73 + 24LL), (__int64)v73, (__int64)&v79, a2, v52);
                v51 = v70;
              }
              v34 = v51;
              v75[1] = v51;
              v29 = *(_BYTE *)(v22 + 23) & 0x40;
            }
            if ( v29 )
              v35 = *(_QWORD *)(v22 - 8);
            else
              v35 = v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
            v36 = (_QWORD *)(v35 + 3 * v24);
            if ( *v36 )
            {
              v37 = v36[1];
              v38 = v36[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v38 = v37;
              if ( v37 )
                *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
            }
            *v36 = v34;
            if ( v34 )
            {
              v39 = v34[1];
              v36[1] = v39;
              if ( v39 )
                *(_QWORD *)(v39 + 16) = (unsigned __int64)(v36 + 1) | *(_QWORD *)(v39 + 16) & 3LL;
              v36[2] = (unsigned __int64)(v34 + 1) | v36[2] & 3LL;
              v34[1] = v36;
            }
            v24 += 8;
            v25 = *((_QWORD *)&v82 + 1);
            if ( v26 == v24 )
              goto LABEL_31;
          }
          *(_QWORD *)&v82 = v82 + 1;
          goto LABEL_68;
        }
        v25 = 0;
LABEL_31:
        j___libc_free_0(v25);
      }
      else
      {
        v79 = sub_1649960(v20);
        LOWORD(v83) = 773;
        v80 = v45;
        *(_QWORD *)&v82 = &v79;
        *((_QWORD *)&v82 + 1) = ".reload";
        v46 = sub_1648A60(64, 1u);
        v47 = (__int64)v46;
        if ( v46 )
          sub_15F90C0((__int64)v46, *(_QWORD *)(*v73 + 24LL), (__int64)v73, (__int64)&v82, a2, v22);
        sub_1648780(v22, v20, v47);
      }
      v19 = *(_QWORD *)(v20 + 8);
      if ( !v19 )
      {
        v11 = v20;
        break;
      }
    }
  }
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 16) - 25 > 9 )
  {
    for ( i = *(_QWORD *)(v11 + 32); ; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      v41 = *(unsigned __int8 *)(i - 8);
      if ( (_BYTE)v41 != 77 )
      {
        v42 = v41 - 34;
        if ( v42 > 0x36 || ((1LL << v42) & 0x40018000000001LL) == 0 )
          break;
      }
    }
    goto LABEL_40;
  }
  i = sub_157EE30(*(_QWORD *)(v11 - 48));
  if ( i )
LABEL_40:
    i -= 24;
  v43 = sub_1648A60(64, 2u);
  if ( v43 )
    sub_15F9660((__int64)v43, v11, (__int64)v73, i);
  return v73;
}
